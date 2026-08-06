#!/usr/bin/env python3
"""Generate the original sparse tomogram with bounded GPU memory.

Spatial tiles include the complete dependency halo of the gradient,
traversability and inflation operations. Layer simplification remains a global
operation: the first pass writes the two required dense fields to temporary
memmaps, and the second pass exports values using the resulting global layer
indices. This preserves the semantics and ordering of tomography.py.
"""

import argparse
import gc
import os
import pickle
import shutil
import string
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import numpy as np
import open3d as o3d

from kernels import inflationKernel, travKernel

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config
from config.scene import ScenePCD


RSG_ROOT = Path(__file__).resolve().parents[2]
PCD_DIR = RSG_ROOT / "rsc" / "pcd"

SPARSE_ASTAR_COST_THRESHOLD = 35.0
SPARSE_GATEWAY_COST_DELTA = 8.0
SPARSE_GATEWAY_HEIGHT_DELTA = 0.1


def log(message):
    print(f"[tiled-tomography] {message}", flush=True)


@dataclass(frozen=True)
class Tile:
    core_x0: int
    core_x1: int
    core_y0: int
    core_y1: int
    data_x0: int
    data_x1: int
    data_y0: int
    data_y1: int

    @property
    def data_dim_x(self):
        return self.data_x1 - self.data_x0

    @property
    def data_dim_y(self):
        return self.data_y1 - self.data_y0

    @property
    def core_local_x(self):
        return slice(self.core_x0 - self.data_x0, self.core_x1 - self.data_x0)

    @property
    def core_local_y(self):
        return slice(self.core_y0 - self.data_y0, self.core_y1 - self.data_y0)


def tiled_tomography_kernel(
    resolution,
    global_dim_x,
    global_dim_y,
    data_x0,
    data_y0,
    data_dim_x,
    data_dim_y,
    n_slice,
    slice_h0,
    slice_dh,
):
    """Map with the original global rounding, then address a local tile."""
    preamble = string.Template(
        r'''
        __device__ int getIndexLine(float x, float center)
        {
            int i = round((x - center) / ${resolution});
            return i;
        }

        __device__ static float atomicMaxFloat(float* address, float val)
        {
            int* address_as_i = (int*) address;
            int old = *address_as_i, assumed;
            do {
                assumed = old;
                old = ::atomicCAS(address_as_i, assumed,
                    __float_as_int(::fmaxf(val, __int_as_float(assumed))));
            } while (assumed != old);
            return __int_as_float(old);
        }

        __device__ static float atomicMinFloat(float* address, float val)
        {
            int* address_as_i = (int*) address;
            int old = *address_as_i, assumed;
            do {
                assumed = old;
                old = ::atomicCAS(address_as_i, assumed,
                    __float_as_int(::fminf(val, __int_as_float(assumed))));
            } while (assumed != old);
            return __int_as_float(old);
        }
        '''
    ).substitute(resolution=resolution)

    operation = string.Template(
        r'''
        U px = points[i * 3];
        U py = points[i * 3 + 1];
        U pz = points[i * 3 + 2];

        int global_x = getIndexLine(px, center[0]) + ${global_dim_x} / 2;
        int global_y = getIndexLine(py, center[1]) + ${global_dim_y} / 2;
        int local_x = global_x - ${data_x0};
        int local_y = global_y - ${data_y0};
        if (local_x < 0 || local_x >= ${data_dim_x} ||
            local_y < 0 || local_y >= ${data_dim_y})
            return;

        int idx = ${data_dim_y} * local_x + local_y;
        for (int s_idx = 0; s_idx < ${n_slice}; s_idx++)
        {
            U slice = ${slice_h0} + s_idx * ${slice_dh};
            int block_idx = ${layer_size} * s_idx + idx;
            if (pz <= slice)
                atomicMaxFloat(&layers_g[block_idx], pz);
            else
                atomicMinFloat(&layers_c[block_idx], pz);
        }
        '''
    ).substitute(
        global_dim_x=global_dim_x,
        global_dim_y=global_dim_y,
        data_x0=data_x0,
        data_y0=data_y0,
        data_dim_x=data_dim_x,
        data_dim_y=data_dim_y,
        n_slice=n_slice,
        slice_h0=slice_h0,
        slice_dh=slice_dh,
        layer_size=data_dim_x * data_dim_y,
    )

    return cp.ElementwiseKernel(
        in_params="raw U points, raw U center",
        out_params="raw U layers_g, raw U layers_c",
        preamble=preamble,
        operation=operation,
        name="tiled_tomography_kernel",
    )


class TileProcessor:
    def __init__(self, scene_cfg, map_dim_x, map_dim_y, n_slice, slice_h0, tile):
        self.resolution = scene_cfg.map.resolution
        self.slice_dh = scene_cfg.map.slice_dh
        self.half_trav = int(scene_cfg.trav.kernel_size / 2)
        self.half_inf = int(
            (scene_cfg.trav.safe_margin + scene_cfg.trav.inflation) / self.resolution
        )
        self.tile = tile
        self.n_slice = n_slice

        shape = (n_slice, tile.data_dim_x, tile.data_dim_y)
        self.layers_g = cp.empty(shape, dtype=cp.float32)
        self.layers_c = cp.empty(shape, dtype=cp.float32)
        self.grad_mag_sq = cp.empty(shape, dtype=cp.float32)
        self.grad_mag_max = cp.empty(shape, dtype=cp.float32)
        self.trav_cost = cp.empty(shape, dtype=cp.float32)
        self.inflated_cost = cp.empty(shape, dtype=cp.float32)

        self.tomography_kernel = tiled_tomography_kernel(
            self.resolution,
            map_dim_x,
            map_dim_y,
            tile.data_x0,
            tile.data_y0,
            tile.data_dim_x,
            tile.data_dim_y,
            n_slice,
            slice_h0,
            self.slice_dh,
        )
        self.trav_kernel = travKernel(
            tile.data_dim_x,
            tile.data_dim_y,
            self.half_trav,
            scene_cfg.trav.interval_min,
            scene_cfg.trav.interval_free,
            scene_cfg.trav.step_max,
            1.2 * self.resolution * np.tan(scene_cfg.trav.slope_max),
            int(
                scene_cfg.trav.standable_ratio * (2 * self.half_trav + 1) ** 2
            )
            - 1,
            float(scene_cfg.trav.cost_barrier),
        )
        self.inflation_kernel = inflationKernel(
            tile.data_dim_x, tile.data_dim_y, self.half_inf
        )

        self.inf_table = cp.zeros(
            (2 * self.half_inf + 1, 2 * self.half_inf + 1), dtype=cp.float32
        )
        for i in range(self.inf_table.shape[0]):
            for j in range(self.inf_table.shape[1]):
                dist = np.sqrt(
                    (self.resolution * (i - self.half_inf)) ** 2
                    + (self.resolution * (j - self.half_inf)) ** 2
                )
                self.inf_table[i, j] = np.clip(
                    1
                    - (dist - scene_cfg.trav.inflation)
                    / (scene_cfg.trav.safe_margin + self.resolution),
                    a_min=0.0,
                    a_max=1.0,
                )

    def compute(self, points, center):
        self.layers_g.fill(np.float32(-1e6))
        self.layers_c.fill(np.float32(1e6))
        self.grad_mag_sq.fill(0)
        self.grad_mag_max.fill(0)
        self.trav_cost.fill(0)
        self.inflated_cost.fill(0)

        self.tomography_kernel(
            points, center, self.layers_g, self.layers_c, size=points.shape[0]
        )

        diff_x_sq = cp.maximum(
            (self.layers_g[:, 1:-1, :] - self.layers_g[:, :-2, :]) ** 2,
            (self.layers_g[:, 1:-1, :] - self.layers_g[:, 2:, :]) ** 2,
        )
        diff_y_sq = cp.maximum(
            (self.layers_g[:, :, 1:-1] - self.layers_g[:, :, :-2]) ** 2,
            (self.layers_g[:, :, 1:-1] - self.layers_g[:, :, 2:]) ** 2,
        )
        self.grad_mag_sq[:, 1:-1, 1:-1] = (
            diff_x_sq[:, :, 1:-1] + diff_y_sq[:, 1:-1, :]
        )
        self.grad_mag_max[:, 1:-1, 1:-1] = cp.maximum(
            diff_x_sq[:, :, 1:-1], diff_y_sq[:, 1:-1, :]
        )
        del diff_x_sq, diff_y_sq

        interval = self.layers_c - self.layers_g
        self.trav_kernel(
            interval,
            self.grad_mag_sq,
            self.grad_mag_max,
            self.trav_cost,
            size=self.n_slice * self.tile.data_dim_x * self.tile.data_dim_y,
        )
        del interval
        self.inflation_kernel(
            self.trav_cost,
            self.inf_table,
            self.inflated_cost,
            size=self.n_slice * self.tile.data_dim_x * self.tile.data_dim_y,
        )
        cp.cuda.get_current_stream().synchronize()
        self.pool_peak_gib = cp.get_default_memory_pool().total_bytes() / 1024**3


def make_tiles(map_dim_x, map_dim_y, n_slice, halo, memory_gb, tile_size, point_bytes):
    if memory_gb <= 0:
        raise ValueError("--gpu-memory-gb must be positive")
    if tile_size is not None:
        if tile_size < 1:
            raise ValueError("--tile-size must be positive")
        core_dim_x = tile_size
        core_dim_y = tile_size
    else:
        budget_bytes = int(memory_gb * 1024**3)
        # The unchanged chained CuPy expressions peak at roughly thirteen
        # float volumes after allocator block rounding (measured on the target
        # GPU). Keep the resulting pool below the requested budget.
        usable_bytes = int(budget_bytes * 0.95) - point_bytes
        bytes_per_cell = 13 * np.dtype(np.float32).itemsize * n_slice
        max_local_cells = usable_bytes // bytes_per_cell
        if max_local_cells <= (2 * halo + 1) ** 2:
            raise ValueError("GPU memory budget is too small for one haloed tile")

        full_y_cells = map_dim_y
        local_dim_x = max_local_cells // full_y_cells
        if local_dim_x > 2 * halo:
            core_dim_x = max(1, int(local_dim_x - 2 * halo))
            core_dim_y = map_dim_y
        else:
            local_side = int(np.sqrt(max_local_cells))
            core_dim_x = max(1, local_side - 2 * halo)
            core_dim_y = core_dim_x

    tiles = []
    for x0 in range(0, map_dim_x, core_dim_x):
        x1 = min(x0 + core_dim_x, map_dim_x)
        for y0 in range(0, map_dim_y, core_dim_y):
            y1 = min(y0 + core_dim_y, map_dim_y)
            tiles.append(
                Tile(
                    x0,
                    x1,
                    y0,
                    y1,
                    max(0, x0 - halo),
                    min(map_dim_x, x1 + halo),
                    max(0, y0 - halo),
                    min(map_dim_y, y1 + halo),
                )
            )
    return tiles


def compute_global_layer_indices(layers_g, inflated_cost, cost_barrier):
    n_slice = layers_g.shape[0]
    indices = [0]
    if n_slice > 1:
        lower_idx = 0
        middle_idx = 1
        while middle_idx < n_slice - 2:
            unique = (
                (
                    (layers_g[middle_idx] - layers_g[lower_idx] > 0)
                    | (inflated_cost[lower_idx] > inflated_cost[middle_idx])
                )
                & (layers_g[middle_idx + 1] - layers_g[middle_idx] > 0)
                & (inflated_cost[middle_idx] < cost_barrier)
            )
            if np.any(unique):
                indices.append(middle_idx)
                lower_idx = middle_idx
            middle_idx += 1
        indices.append(middle_idx)
    return np.asarray(indices, dtype=np.int32)


def compute_gateway(trav, elev_g):
    diff_t = trav[1:] - trav[:-1]
    diff_g = np.abs(elev_g[1:] - elev_g[:-1])

    gateway_up = np.zeros_like(trav, dtype=bool)
    mask_t = diff_t < -SPARSE_GATEWAY_COST_DELTA
    mask_g = (diff_g < SPARSE_GATEWAY_HEIGHT_DELTA) & np.isfinite(elev_g[1:])
    gateway_up[:-1] = mask_t & mask_g

    gateway_dn = np.zeros_like(trav, dtype=bool)
    mask_t = diff_t > SPARSE_GATEWAY_COST_DELTA
    mask_g = (diff_g < SPARSE_GATEWAY_HEIGHT_DELTA) & np.isfinite(elev_g[:-1])
    gateway_dn[1:] = mask_t & mask_g

    gateway = np.zeros_like(trav, dtype=np.int32)
    gateway[gateway_up] = 2
    gateway[gateway_dn] = -2
    return gateway


def release_processor(processor):
    for name in (
        "layers_g",
        "layers_c",
        "grad_mag_sq",
        "grad_mag_max",
        "trav_cost",
        "inflated_cost",
        "inf_table",
        "tomography_kernel",
        "trav_kernel",
        "inflation_kernel",
    ):
        if hasattr(processor, name):
            delattr(processor, name)
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()


def process_tile(scene_cfg, metadata, tile, points_gpu, center_gpu):
    processor = TileProcessor(
        scene_cfg,
        metadata["map_dim_x"],
        metadata["map_dim_y"],
        metadata["n_slice_init"],
        metadata["slice_h0"],
        tile,
    )
    processor.compute(points_gpu, center_gpu)
    return processor


def generate(scene_cfg, pcd_path, output_path, memory_gb, tile_size, temp_dir):
    log(f"loading PCD: {pcd_path}")
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    points = np.asarray(pcd.points, dtype=np.float32)
    points = points[~np.isnan(points).any(axis=1)]
    if points.size == 0:
        raise RuntimeError("loaded an empty point cloud")

    points_max = np.max(points, axis=0)
    points_min = np.min(points, axis=0)
    points_min[-1] = scene_cfg.map.ground_h
    map_dim_x = int(np.ceil((points_max[0] - points_min[0]) / scene_cfg.map.resolution)) + 4
    map_dim_y = int(np.ceil((points_max[1] - points_min[1]) / scene_cfg.map.resolution)) + 4
    n_slice_init = int(
        np.ceil((points_max[2] - points_min[2]) / scene_cfg.map.slice_dh)
    )
    center = (points_max[:2] + points_min[:2]) / 2
    slice_h0 = float(points_min[-1] + scene_cfg.map.slice_dh)

    half_trav = int(scene_cfg.trav.kernel_size / 2)
    half_inf = int(
        (scene_cfg.trav.safe_margin + scene_cfg.trav.inflation)
        / scene_cfg.map.resolution
    )
    halo = 1 + half_trav + half_inf

    metadata = {
        "center": center,
        "map_dim_x": map_dim_x,
        "map_dim_y": map_dim_y,
        "n_slice_init": n_slice_init,
        "slice_h0": slice_h0,
    }

    cp.get_default_memory_pool().free_all_blocks()
    points_gpu = cp.asarray(points)
    center_gpu = cp.asarray(center, dtype=cp.float32)
    tiles = make_tiles(
        map_dim_x,
        map_dim_y,
        n_slice_init,
        halo,
        memory_gb,
        tile_size,
        points_gpu.nbytes,
    )

    log(f"points={points.shape[0]:,}")
    log(f"map=[{n_slice_init}, {map_dim_x}, {map_dim_y}], halo={halo}")
    log(f"tiles={len(tiles)}, memory budget={memory_gb:.2f} GiB")

    temp_parent = Path(temp_dir).resolve() if temp_dir else output_path.parent
    temp_parent.mkdir(parents=True, exist_ok=True)
    memmap_bytes = 2 * n_slice_init * map_dim_x * map_dim_y * np.dtype(np.float32).itemsize
    if shutil.disk_usage(temp_parent).free < memmap_bytes + 512 * 1024**2:
        raise RuntimeError(
            f"not enough temporary disk space in {temp_parent}; "
            f"need at least {(memmap_bytes + 512 * 1024**2) / 1024**3:.2f} GiB"
        )
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix=".tiled_tomography.", dir=temp_parent) as work_dir:
        work_dir = Path(work_dir)
        dense_shape = (n_slice_init, map_dim_x, map_dim_y)
        layers_g_map = np.memmap(
            work_dir / "layers_g.dat", mode="w+", dtype=np.float32, shape=dense_shape
        )
        inflated_map = np.memmap(
            work_dir / "inflated_cost.dat",
            mode="w+",
            dtype=np.float32,
            shape=dense_shape,
        )

        log("pass 1/2: computing global layer simplification inputs")
        for tile_idx, tile in enumerate(tiles, start=1):
            processor = process_tile(scene_cfg, metadata, tile, points_gpu, center_gpu)
            local_x = tile.core_local_x
            local_y = tile.core_local_y
            layers_g_map[
                :, tile.core_x0 : tile.core_x1, tile.core_y0 : tile.core_y1
            ] = processor.layers_g[:, local_x, local_y].get()
            inflated_map[
                :, tile.core_x0 : tile.core_x1, tile.core_y0 : tile.core_y1
            ] = processor.inflated_cost[:, local_x, local_y].get()
            log(
                f"pass 1 tile {tile_idx}/{len(tiles)}: "
                f"x={tile.core_x0}:{tile.core_x1}, y={tile.core_y0}:{tile.core_y1}, "
                f"GPU pool peak={processor.pool_peak_gib:.2f} GiB"
            )
            release_processor(processor)

        layers_g_map.flush()
        inflated_map.flush()
        layer_indices = compute_global_layer_indices(
            layers_g_map, inflated_map, float(scene_cfg.trav.cost_barrier)
        )
        log(
            f"global layer simplification: {n_slice_init} -> {len(layer_indices)}; "
            f"indices={layer_indices.tolist()}"
        )
        del layers_g_map, inflated_map

        sparse_parts = {name: [] for name in ("indices", "trav", "elev_g", "elev_c", "gateway")}
        log("pass 2/2: exporting globally simplified sparse values")
        for tile_idx, tile in enumerate(tiles, start=1):
            processor = process_tile(scene_cfg, metadata, tile, points_gpu, center_gpu)
            local_x = tile.core_local_x
            local_y = tile.core_local_y
            local_sel = (layer_indices, local_x, local_y)
            layers_t = processor.inflated_cost[local_sel].get()
            layers_g_raw = processor.layers_g[local_sel].get()
            layers_c_raw = processor.layers_c[local_sel].get()
            layers_g = np.where(layers_g_raw > -1e6, layers_g_raw, np.nan)
            layers_c = np.where(layers_c_raw < 1e6, layers_c_raw, np.nan)

            gateway = compute_gateway(layers_t, layers_g)
            sparse_mask = (
                np.isfinite(layers_g)
                & np.isfinite(layers_t)
                & (layers_t <= SPARSE_ASTAR_COST_THRESHOLD)
            ) | (gateway != 0)
            coords = np.argwhere(sparse_mask)
            if coords.size:
                indices = np.stack(
                    [
                        coords[:, 0],
                        coords[:, 2] + tile.core_y0,
                        coords[:, 1] + tile.core_x0,
                    ],
                    axis=1,
                ).astype(np.int32)
                sparse_parts["indices"].append(indices)
                sparse_parts["trav"].append(layers_t[sparse_mask].astype(np.float32))
                sparse_parts["elev_g"].append(layers_g[sparse_mask].astype(np.float32))
                sparse_parts["elev_c"].append(layers_c[sparse_mask].astype(np.float32))
                sparse_parts["gateway"].append(gateway[sparse_mask].astype(np.int32))

            log(
                f"pass 2 tile {tile_idx}/{len(tiles)}: "
                f"x={tile.core_x0}:{tile.core_x1}, y={tile.core_y0}:{tile.core_y1}, "
                f"nodes={int(coords.shape[0]):,}"
            )
            release_processor(processor)

    del points_gpu, center_gpu
    cp.get_default_memory_pool().free_all_blocks()

    if sparse_parts["indices"]:
        indices = np.concatenate(sparse_parts["indices"], axis=0)
        trav = np.concatenate(sparse_parts["trav"], axis=0)
        elev_g = np.concatenate(sparse_parts["elev_g"], axis=0)
        elev_c = np.concatenate(sparse_parts["elev_c"], axis=0)
        gateway = np.concatenate(sparse_parts["gateway"], axis=0)

        # np.argwhere on the original [layer, x, y] array orders by layer, x, y.
        order = np.lexsort((indices[:, 1], indices[:, 2], indices[:, 0]))
        indices = indices[order]
        trav = trav[order]
        elev_g = elev_g[order]
        elev_c = elev_c[order]
        gateway = gateway[order]
    else:
        indices = np.empty((0, 3), dtype=np.int32)
        trav = np.empty(0, dtype=np.float32)
        elev_g = np.empty(0, dtype=np.float32)
        elev_c = np.empty(0, dtype=np.float32)
        gateway = np.empty(0, dtype=np.int32)

    slice_heights = (
        slice_h0
        + cp.arange(n_slice_init, dtype=cp.float32) * scene_cfg.map.slice_dh
    ).get()[layer_indices]
    cp.get_default_memory_pool().free_all_blocks()

    data_dict = {
        "format": "tomogram_sparse_v1",
        "shape": [int(len(layer_indices)), int(map_dim_y), int(map_dim_x)],
        "resolution": float(scene_cfg.map.resolution),
        "center": center.astype(np.float64),
        "slice_h0": slice_h0,
        "slice_dh": float(scene_cfg.map.slice_dh),
        "slice_heights": slice_heights.astype(np.float32),
        "indices": indices,
        "trav": trav,
        "elev_g": elev_g,
        "elev_c": elev_c,
        "gateway": gateway,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    dense_cells = len(layer_indices) * map_dim_y * map_dim_x
    log(f"sparse nodes={indices.shape[0]:,}")
    log(f"dense cells={dense_cells:,}")
    log(f"elapsed={time.time() - t0:.2f}s")
    print(f"Sparse tomogram exported: {output_path.name}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="Map")
    parser.add_argument("--pcd", required=True, help="PCD basename under rsc/pcd")
    parser.add_argument(
        "--output",
        default=str(RSG_ROOT / "rsc" / "tomogram" / "scene_map_sparse.pickle"),
    )
    parser.add_argument(
        "--gpu-memory-gb",
        type=float,
        default=10.0,
        help="GPU allocation budget used to derive automatic tile dimensions",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="force square core tile size; intended for equivalence tests",
    )
    parser.add_argument("--temp-dir", default=None, help="directory for temporary dense memmaps")
    return parser.parse_args()


def main():
    args = parse_args()
    scene_cfg = getattr(config, "Scene" + args.scene)()
    if not hasattr(scene_cfg, "pcd"):
        scene_cfg.pcd = ScenePCD()
    scene_cfg.pcd.file_name = args.pcd

    pcd_path = Path(args.pcd).expanduser()
    if not pcd_path.is_absolute():
        pcd_path = PCD_DIR / pcd_path
    output_path = Path(args.output).expanduser().resolve()
    generate(
        scene_cfg,
        pcd_path.resolve(),
        output_path,
        args.gpu_memory_gb,
        args.tile_size,
        args.temp_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

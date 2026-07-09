#!/usr/bin/env python3
"""Extract the largest physically connected component from a sparse tomogram pickle.

Physical connectivity means:
- each height slice is labeled with 2-D connectivity (4/8-neighborhood)
- adjacent slices are bridged when enough overlapping cells satisfy
  |elev_g difference| <= max_vertical_step within connect_z_xy XY radius
- the largest bridged component is kept

Input format: tomogram_sparse_v1 (the format produced by tomography.py).
Output format: same sparse v1 pickle with only the largest component retained.
"""

import argparse
import os
import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np
from scipy import ndimage


RSG_ROOT = Path(__file__).resolve().parents[2]


def log(message: str) -> None:
    print(f"[extract-physical] {message}", flush=True)


def resolve_input(path: str) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    if p.exists():
        return p.resolve()
    name = path if path.endswith(".pickle") else path + ".pickle"
    return (RSG_ROOT / "rsc" / "tomogram" / name).resolve()


def resolve_output(path: str | None, input_path: Path) -> Path:
    if not path:
        return input_path.with_name(f"{input_path.stem}_physical{input_path.suffix}")
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (input_path.parent / p).resolve()


def load_sparse_pickle(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"pickle not found: {path}")
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise ValueError("expected dict")
    if data.get("format") != "tomogram_sparse_v1":
        raise ValueError(f"expected format 'tomogram_sparse_v1', got {data.get('format')!r}")
    required = {"shape", "indices", "trav", "elev_g", "elev_c", "gateway"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"sparse pickle missing keys: {missing}")
    return data


def build_structure_2d(connect_xy: int) -> np.ndarray:
    if connect_xy == 4:
        return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    if connect_xy == 8:
        return np.ones((3, 3), dtype=bool)
    raise ValueError("--connect-xy must be 4 or 8")


def shifted_slices(dim_x: int, dim_y: int, dx: int, dy: int):
    prev_x0, prev_x1 = max(0, -dx), min(dim_x, dim_x - dx)
    prev_y0, prev_y1 = max(0, -dy), min(dim_y, dim_y - dy)
    if prev_x0 >= prev_x1 or prev_y0 >= prev_y1:
        return None
    return (
        (slice(prev_x0, prev_x1), slice(prev_y0, prev_y1)),
        (slice(prev_x0 + dx, prev_x1 + dx), slice(prev_y0 + dy, prev_y1 + dy)),
    )


def uf_find(parent: np.ndarray, item: int) -> int:
    root = item
    while parent[root] != root:
        root = int(parent[root])
    while parent[item] != item:
        nxt = int(parent[item])
        parent[item] = root
        item = nxt
    return root


def uf_union(parent: np.ndarray, rank: np.ndarray, a: int, b: int) -> None:
    ra, rb = uf_find(parent, a), uf_find(parent, b)
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        parent[ra] = rb
    elif rank[ra] > rank[rb]:
        parent[rb] = ra
    else:
        parent[rb] = ra
        rank[ra] += 1


def extract_largest_physical_component(
    traversable: np.ndarray,
    layers_g: np.ndarray,
    connect_xy: int,
    connect_z_xy: int,
    max_vertical_step: float,
    min_bridge_cells: int,
):
    """Return a bool mask of the largest physically connected component."""
    structure2d = build_structure_2d(connect_xy)
    n_slices, dim_x, dim_y = traversable.shape

    slice_offsets = []
    slice_counts = []
    node_sizes = []
    edge_a = []
    edge_b = []

    prev_labels = None
    prev_offset = 0
    prev_count = 0
    prev_g = None
    current_offset = 0

    shifts = [
        (dx, dy)
        for dx in range(-connect_z_xy, connect_z_xy + 1)
        for dy in range(-connect_z_xy, connect_z_xy + 1)
        if abs(dx) + abs(dy) <= connect_z_xy
    ]

    for s in range(n_slices):
        labels, count = ndimage.label(traversable[s], structure=structure2d)
        labels = labels.astype(np.int32, copy=False)
        slice_offsets.append(current_offset)
        slice_counts.append(int(count))

        if count > 0:
            node_sizes.extend(
                np.bincount(labels.ravel(), minlength=count + 1)[1:].astype(np.int64).tolist()
            )

        if prev_labels is not None and prev_count > 0 and count > 0:
            curr_g = layers_g[s].astype(np.float32, copy=False)
            pair_base = count + 1
            for dx, dy in shifts:
                slices = shifted_slices(dim_x, dim_y, dx, dy)
                if slices is None:
                    continue
                prev_slice, curr_slice = slices
                prev_lab = prev_labels[prev_slice]
                curr_lab = labels[curr_slice]
                cand = (prev_lab > 0) & (curr_lab > 0)
                if not np.any(cand):
                    continue
                dz = np.abs(prev_g[prev_slice] - curr_g[curr_slice])
                cand &= dz <= max_vertical_step
                if not np.any(cand):
                    continue

                encoded = (
                    prev_lab[cand].astype(np.int64) * pair_base
                    + curr_lab[cand].astype(np.int64)
                )
                if min_bridge_cells > 1:
                    pairs, counts = np.unique(encoded, return_counts=True)
                    pairs = pairs[counts >= min_bridge_cells]
                else:
                    pairs = np.unique(encoded)
                if pairs.size == 0:
                    continue

                edge_a.extend((prev_offset + pairs // pair_base - 1).astype(np.int64).tolist())
                edge_b.extend((current_offset + pairs % pair_base - 1).astype(np.int64).tolist())

        prev_labels = labels
        prev_offset = current_offset
        prev_count = int(count)
        prev_g = layers_g[s].astype(np.float32, copy=False)
        current_offset += int(count)

    if current_offset == 0:
        raise RuntimeError("no traversable cells found")

    parent = np.arange(current_offset, dtype=np.int32)
    rank = np.zeros(current_offset, dtype=np.int8)
    for a, b in zip(edge_a, edge_b):
        uf_union(parent, rank, int(a), int(b))

    roots = np.array([uf_find(parent, i) for i in range(current_offset)], dtype=np.int32)
    sizes = np.bincount(roots, weights=np.asarray(node_sizes, dtype=np.int64), minlength=current_offset)
    largest_root = int(np.argmax(sizes))
    selected = roots == largest_root

    keep = np.zeros(traversable.shape, dtype=bool)
    for s in range(n_slices):
        count = slice_counts[s]
        if count == 0:
            continue
        offset = slice_offsets[s]
        sel = selected[offset : offset + count]
        if not np.any(sel):
            continue
        labels, _ = ndimage.label(traversable[s], structure=structure2d)
        pos = labels > 0
        keep[s][pos] = sel[labels[pos] - 1]

    return keep, int(sizes[largest_root]), current_offset, len(edge_a)


def write_pickle_atomic(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(path)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep only the largest physically connected component of a sparse tomogram."
    )
    parser.add_argument("--pickle", type=str, default="scene_map_sparse.pickle", help="input sparse tomogram pickle")
    parser.add_argument("--output", type=str, default=None, help="output pickle; defaults to <input>_physical.pickle")
    parser.add_argument("--cost-threshold", type=float, default=35.0, help="traversable nodes have trav <= this")
    parser.add_argument("--connect-xy", type=int, choices=(4, 8), default=8, help="same-layer connectivity")
    parser.add_argument("--connect-z-xy", type=int, default=1, help="XY manhattan radius for adjacent-slice bridges")
    parser.add_argument("--max-vertical-step", type=float, default=0.75, help="max |elev_g| difference for a bridge")
    parser.add_argument("--min-bridge-cells", type=int, default=1, help="min overlapping cells to create a bridge")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = resolve_input(args.pickle)
    output_path = resolve_output(args.output, input_path)
    if output_path.is_dir():
        raise ValueError(f"--output must be a file path, not a directory: {output_path}")

    log(f"loading {input_path}")
    data_dict = load_sparse_pickle(input_path)

    indices = np.asarray(data_dict["indices"], dtype=np.int32)
    trav = np.asarray(data_dict["trav"], dtype=np.float32)
    elev_g = np.asarray(data_dict["elev_g"], dtype=np.float32)
    gateway = np.asarray(data_dict["gateway"], dtype=np.int32)
    shape = tuple(int(x) for x in data_dict["shape"])  # [S, Y, X]

    # Reconstruct dense arrays from the sparse representation.
    traversable_mask = np.zeros(shape, dtype=bool)
    layers_g = np.full(shape, np.nan, dtype=np.float32)
    layers_t = np.full(shape, np.nan, dtype=np.float32)
    layers_gateway = np.zeros(shape, dtype=np.int32)
    traversable_mask[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    layers_g[indices[:, 0], indices[:, 1], indices[:, 2]] = elev_g
    layers_t[indices[:, 0], indices[:, 1], indices[:, 2]] = trav
    layers_gateway[indices[:, 0], indices[:, 1], indices[:, 2]] = gateway

    traversable = (
        traversable_mask
        & np.isfinite(layers_g)
        & np.isfinite(layers_t)
        & ((layers_t <= float(args.cost_threshold)) | (layers_gateway != 0))
    )
    total = int(np.count_nonzero(traversable))
    log(f"traversable nodes={total:,}")
    if total == 0:
        raise RuntimeError("no traversable nodes")

    log(
        f"extracting physical component: connect_xy={args.connect_xy}, "
        f"connect_z_xy={args.connect_z_xy}, max_vertical_step={args.max_vertical_step:.3f}, "
        f"min_bridge_cells={args.min_bridge_cells}"
    )
    keep, largest_count, n_nodes, n_edges = extract_largest_physical_component(
        traversable,
        layers_g,
        args.connect_xy,
        args.connect_z_xy,
        float(args.max_vertical_step),
        int(args.min_bridge_cells),
    )
    log(
        f"largest component: nodes={largest_count:,}, ratio={largest_count / total:.4f}, "
        f"graph_nodes={n_nodes:,}, graph_edges={n_edges:,}"
    )

    keep_idx = keep[indices[:, 0], indices[:, 1], indices[:, 2]]
    removed = int(np.count_nonzero(~keep_idx))
    log(f"removed nodes={removed:,}")

    out = dict(data_dict)
    out["indices"] = indices[keep_idx]
    out["trav"] = trav[keep_idx]
    out["elev_g"] = elev_g[keep_idx]
    out["elev_c"] = np.asarray(data_dict["elev_c"], dtype=np.float32)[keep_idx]
    out["gateway"] = gateway[keep_idx]
    out["connectivity_source"] = {
        "input": str(input_path),
        "method": "physical",
        "cost_threshold": float(args.cost_threshold),
        "connect_xy": int(args.connect_xy),
        "connect_z_xy": int(args.connect_z_xy),
        "max_vertical_step": float(args.max_vertical_step),
        "min_bridge_cells": int(args.min_bridge_cells),
        "total_nodes": int(indices.shape[0]),
        "total_traversable_nodes": total,
        "largest_nodes": largest_count,
        "removed_nodes": removed,
        "graph_nodes": n_nodes,
        "graph_edges": n_edges,
    }

    input_size = os.path.getsize(input_path)
    free = shutil.disk_usage(output_path.parent).free
    log(f"output nodes={int(out['indices'].shape[0]):,}")
    if free < input_size * 1.2 + 512 * 1024 * 1024:
        raise RuntimeError(f"not enough free disk space on {output_path.parent}")

    log(f"writing {output_path}")
    write_pickle_atomic(out, output_path)
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

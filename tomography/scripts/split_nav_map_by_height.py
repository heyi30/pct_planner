#!/usr/bin/env python3
"""Split a sparse nav_map into height-layer regions and export each bbox as JSON.

Workflow:
1. Load a sparse tomogram pickle (the nav_map).
2. Keep only traversable / gateway nodes.
3. Compute the local ground slope from elev_g.
4. Use the slope mask as the boundary to label connected regions at similar
   height on each layer ("以斜率的形式识别出处在相似高度的大面积联通区域").
5. Keep only large components (> --min-nodes). Layer height is not limited:
   each slope-bounded component is one region.
6. Compute an axis-aligned world-frame bbox for each region and write the
   result as JSON.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from scipy import ndimage


RSG_ROOT = Path(__file__).resolve().parents[2]
SPARSE_ASTAR_COST_THRESHOLD = 35.0


def log(message: str) -> None:
    print(f"[split-nav-by-height] {message}", flush=True)


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
        return input_path.with_name(f"{input_path.stem}_height_regions.json")
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve()


def load_sparse_pickle(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"pickle not found: {path}")
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise ValueError("expected dict")
    if data.get("format") != "tomogram_sparse_v1":
        raise ValueError(f"expected format 'tomogram_sparse_v1', got {data.get('format')!r}")
    required = {"shape", "indices", "trav", "elev_g", "elev_c", "gateway", "resolution", "center"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"sparse pickle missing keys: {missing}")
    return data


def build_dense_height(
    shape: tuple[int, int, int],
    indices: np.ndarray,
    elev_g: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    """Return a dense height array (S, Y, X) with NaN for inactive cells."""
    height = np.full(shape, np.nan, dtype=np.float32)
    height[indices[active, 0], indices[active, 1], indices[active, 2]] = elev_g[active]
    return height


def compute_slope(height: np.ndarray, resolution: float) -> np.ndarray:
    """Return the magnitude of the ground slope (tan theta) for each cell.

    Boundaries are handled by np.gradient; inactive NaN cells are filled with a
    large slope so they are excluded from the flat mask.
    """
    # np.gradient gives dz/d(axis index) for the spatial axes. Convert to physical slope.
    dz_dy, dz_dx = np.gradient(height, axis=(1, 2))
    slope = np.hypot(dz_dx / resolution, dz_dy / resolution)
    # Keep NaN cells as NaN so they are excluded from flat regions.
    slope[~np.isfinite(height)] = np.nan
    return slope


def label_flat_components(
    height: np.ndarray,
    slope: np.ndarray,
    slope_max: float,
    min_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Label connected flat-ish components per layer.

    Returns:
        labels: dense int32 array (S, Y, X) with globally unique component ids,
                -1 for background.
        sizes: 1-D array indexed by component id giving node counts.
    """
    n_slice, n_y, n_x = height.shape
    flat_mask = np.isfinite(slope) & (slope <= slope_max)

    labels = np.full((n_slice, n_y, n_x), -1, dtype=np.int32)
    next_label = 0
    all_sizes: list[np.ndarray] = []

    # 8-connectivity in the layer plane.
    structure = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)

    for s in range(n_slice):
        if not np.any(flat_mask[s]):
            continue
        layer_labels, n_feat = ndimage.label(flat_mask[s], structure=structure)
        if n_feat == 0:
            continue
        # ndimage.label produces 1..n_feat. Convert to 0-based, then offset
        # so labels are globally unique and sizes[global_id] is its node count.
        layer_labels_0 = layer_labels - 1
        valid = layer_labels > 0
        labels[s][valid] = layer_labels_0[valid] + next_label
        sizes = np.bincount(layer_labels_0[valid].ravel(), minlength=n_feat)
        all_sizes.append(sizes)
        next_label += n_feat

    if not all_sizes:
        return labels, np.array([], dtype=np.int64)

    full_sizes = np.concatenate(all_sizes)
    return labels, full_sizes


def compute_component_region(
    component_id: int,
    node_ids: np.ndarray,
    coords: np.ndarray,
    elev_g: np.ndarray,
    resolution: float,
    center: np.ndarray,
    shape: tuple[int, int, int],
) -> dict | None:
    """Compute the bbox region for one slope-bounded component.

    Layer height is not limited: the whole component is one region.
    """
    if node_ids.size == 0:
        return None

    z = elev_g[node_ids]
    layers = coords[:, 0]
    rows = coords[:, 1]
    cols = coords[:, 2]

    n_x = shape[2]
    n_y = shape[1]
    cx, cy = center[0], center[1]

    x_world = (cols - 0.5 * n_x) * resolution + cx
    y_world = (rows - 0.5 * n_y) * resolution + cy

    return {
        "component_id": int(component_id),
        "node_count": int(coords.shape[0]),
        "height_mean": float(np.mean(z)),
        "height_min": float(np.min(z)),
        "height_max": float(np.max(z)),
        "height_range": float(np.max(z) - np.min(z)),
        "bbox": {
            "min_x": float(np.min(x_world)),
            "max_x": float(np.max(x_world)),
            "min_y": float(np.min(y_world)),
            "max_y": float(np.max(y_world)),
            "min_z": float(np.min(z)),
            "max_z": float(np.max(z)),
        },
        "center": {
            "x": float(np.mean(x_world)),
            "y": float(np.mean(y_world)),
            "z": float(np.mean(z)),
        },
        "size": {
            "x": float(np.max(x_world) - np.min(x_world) + resolution),
            "y": float(np.max(y_world) - np.min(y_world) + resolution),
            "z": float(np.max(z) - np.min(z)),
        },
        "slices": sorted({int(s) for s in layers.tolist()}),
    }


def compute_iou_3d(a: dict, b: dict) -> float:
    """Compute 3D IoU between two axis-aligned bboxes."""
    ba, bb = a["bbox"], b["bbox"]

    ix = max(0.0, min(ba["max_x"], bb["max_x"]) - max(ba["min_x"], bb["min_x"]))
    iy = max(0.0, min(ba["max_y"], bb["max_y"]) - max(ba["min_y"], bb["min_y"]))
    iz = max(0.0, min(ba["max_z"], bb["max_z"]) - max(ba["min_z"], bb["min_z"]))
    inter = ix * iy * iz
    if inter <= 0.0:
        return 0.0

    vol_a = (ba["max_x"] - ba["min_x"]) * (ba["max_y"] - ba["min_y"]) * (ba["max_z"] - ba["min_z"])
    vol_b = (bb["max_x"] - bb["min_x"]) * (bb["max_y"] - bb["min_y"]) * (bb["max_z"] - bb["min_z"])
    union = vol_a + vol_b - inter
    return inter / union if union > 0.0 else 0.0


def deduplicate_regions(regions: list[dict], iou_threshold: float) -> list[dict]:
    """Remove smaller regions whose 3D bbox overlaps a larger kept region.

    Greedy: sort by node_count descending, keep a region if it does not overlap
    (3D IoU > threshold) with any already-kept region. threshold=0 means any
    overlap is forbidden.
    """
    if iou_threshold < 0.0 or len(regions) <= 1:
        return list(regions)

    sorted_regions = sorted(regions, key=lambda r: r["node_count"], reverse=True)
    kept: list[dict] = []
    removed = 0
    for region in sorted_regions:
        overlap = False
        for kept_region in kept:
            if compute_iou_3d(region, kept_region) > iou_threshold:
                overlap = True
                removed += 1
                log(
                    f"dedup: remove region {region['id']} (nodes={region['node_count']:,}) "
                    f"overlaps region {kept_region['id']} (nodes={kept_region['node_count']:,})"
                )
                break
        if not overlap:
            kept.append(region)

    # Re-assign sequential ids after deduplication.
    for new_id, region in enumerate(kept):
        region["id"] = new_id
    log(f"deduplication: {len(regions)} -> {len(kept)} regions (removed {removed})")
    return kept


class UnionFind:
    """Simple union-find for merging regions by z-gap."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def z_nearest_distance(a: dict, b: dict) -> float:
    """Return the shortest distance between two bboxes along the z axis."""
    ba, bb = a["bbox"], b["bbox"]
    if ba["max_z"] < bb["min_z"]:
        return bb["min_z"] - ba["max_z"]
    if bb["max_z"] < ba["min_z"]:
        return ba["min_z"] - bb["max_z"]
    return 0.0


def merge_regions_by_z_gap(regions: list[dict], z_gap: float) -> list[dict]:
    """Merge regions whose z-direction nearest distance is smaller than z_gap.

    Transitive merging is performed via union-find. Each merged group becomes a
    single region whose bbox is the union of all member bboxes.
    """
    if z_gap < 0.0 or len(regions) <= 1:
        return list(regions)

    n = len(regions)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if z_nearest_distance(regions[i], regions[j]) < z_gap:
                uf.union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = uf.find(i)
        groups.setdefault(root, []).append(i)

    merged: list[dict] = []
    for root, members in groups.items():
        if len(members) == 1:
            merged.append(regions[members[0]])
            continue

        group_regions = [regions[i] for i in members]
        bboxes = [r["bbox"] for r in group_regions]
        min_x = min(b["min_x"] for b in bboxes)
        max_x = max(b["max_x"] for b in bboxes)
        min_y = min(b["min_y"] for b in bboxes)
        max_y = max(b["max_y"] for b in bboxes)
        min_z = min(b["min_z"] for b in bboxes)
        max_z = max(b["max_z"] for b in bboxes)

        total_nodes = sum(r["node_count"] for r in group_regions)
        cx = sum(r["center"]["x"] * r["node_count"] for r in group_regions) / total_nodes
        cy = sum(r["center"]["y"] * r["node_count"] for r in group_regions) / total_nodes
        cz = sum(r["center"]["z"] * r["node_count"] for r in group_regions) / total_nodes

        slices: set[int] = set()
        for r in group_regions:
            slices.update(r["slices"])

        merged.append(
            {
                "component_id": [r["component_id"] for r in group_regions],
                "node_count": total_nodes,
                "height_mean": float(cz),
                "height_min": float(min_z),
                "height_max": float(max_z),
                "height_range": float(max_z - min_z),
                "bbox": {
                    "min_x": float(min_x),
                    "max_x": float(max_x),
                    "min_y": float(min_y),
                    "max_y": float(max_y),
                    "min_z": float(min_z),
                    "max_z": float(max_z),
                },
                "center": {"x": float(cx), "y": float(cy), "z": float(cz)},
                "size": {
                    "x": float(max_x - min_x),
                    "y": float(max_y - min_y),
                    "z": float(max_z - min_z),
                },
                "slices": sorted(slices),
            }
        )
        log(
            f"merge: group {len(merged) - 1} <- {len(members)} regions, "
            f"nodes={total_nodes:,}, z=[{min_z:.3f}, {max_z:.3f}]"
        )

    # Re-assign sequential ids.
    for new_id, region in enumerate(merged):
        region["id"] = new_id
    log(f"z-gap merge: {len(regions)} -> {len(merged)} regions")
    return merged


def split_nav_map(
    data: dict,
    cost_threshold: float,
    slope_max: float,
    min_nodes: int,
    dedup_iou: float | None,
    merge_z_gap: float | None,
) -> dict:
    t0 = time.time()

    indices = np.asarray(data["indices"], dtype=np.int32)
    elev_g = np.asarray(data["elev_g"], dtype=np.float32)
    trav = np.asarray(data["trav"], dtype=np.float32)
    gateway = np.asarray(data["gateway"], dtype=np.int32)
    shape = tuple(int(x) for x in data["shape"])
    resolution = float(data["resolution"])
    center = np.asarray(data["center"], dtype=np.float64)

    traversable = (trav <= cost_threshold) | (gateway != 0)
    n_nodes_total = indices.shape[0]
    n_active = int(np.count_nonzero(traversable))
    log(f"total nodes={n_nodes_total:,}, traversable/gateway={n_active:,}")

    height = build_dense_height(shape, indices, elev_g, traversable)
    log("computing slope ...")
    slope = compute_slope(height, resolution)

    log(f"labeling flat components (slope_max={slope_max}) ...")
    labels_dense, sizes = label_flat_components(height, slope, slope_max, min_nodes)

    n_components = int(sizes.shape[0])
    log(f"flat components={n_components:,}")
    if n_components == 0:
        return {"regions": []}

    large_mask = sizes >= min_nodes
    large_ids = np.flatnonzero(large_mask)
    log(f"large components (>={min_nodes} nodes)={large_ids.size:,}")
    if large_ids.size == 0:
        return {"regions": []}

    # Gather sparse node labels for the active nodes only.
    node_labels = labels_dense[indices[:, 0], indices[:, 1], indices[:, 2]]

    all_regions: list[dict] = []
    for comp_id in large_ids:
        comp_nodes = np.flatnonzero(node_labels == comp_id)
        comp_size = int(comp_nodes.size)
        if comp_size < min_nodes:
            continue
        region = compute_component_region(
            component_id=int(comp_id),
            node_ids=comp_nodes,
            coords=indices[comp_nodes],
            elev_g=elev_g,
            resolution=resolution,
            center=center,
            shape=shape,
        )
        if region is None:
            continue
        region["id"] = len(all_regions)
        all_regions.append(region)
        log(f"component {comp_id}: nodes={comp_size:,}, height_range={region['height_range']:.3f}")

    if dedup_iou is not None:
        log(f"deduplicating regions (3D IoU > {dedup_iou}) ...")
        n_before_dedup = len(all_regions)
        all_regions = deduplicate_regions(all_regions, dedup_iou)
        n_after_dedup = len(all_regions)
    else:
        n_before_dedup = n_after_dedup = len(all_regions)

    if merge_z_gap is not None:
        log(f"merging regions with z-gap < {merge_z_gap}m ...")
        n_before_merge = len(all_regions)
        all_regions = merge_regions_by_z_gap(all_regions, merge_z_gap)
        n_after_merge = len(all_regions)
    else:
        n_before_merge = n_after_merge = len(all_regions)

    elapsed = time.time() - t0
    log(f"generated {len(all_regions)} regions in {elapsed:.2f}s")

    return {
        "regions": all_regions,
        "summary": {
            "total_nodes": n_nodes_total,
            "active_nodes": n_active,
            "flat_components": n_components,
            "large_components": int(large_ids.size),
            "regions_before_dedup": n_before_dedup,
            "regions_after_dedup": n_after_dedup,
            "regions_before_merge": n_before_merge,
            "regions_after_merge": n_after_merge,
            "output_regions": len(all_regions),
            "elapsed_seconds": elapsed,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a sparse nav_map into height-layer regions and export bboxes as JSON."
    )
    parser.add_argument(
        "--pickle",
        type=str,
        default="scene_map_sparse_planner.pickle",
        help="input sparse tomogram pickle (absolute path or basename under rsc/tomogram/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="output JSON path; defaults to <input_stem>_height_regions.json",
    )
    parser.add_argument(
        "--cost-threshold",
        type=float,
        default=SPARSE_ASTAR_COST_THRESHOLD,
        help="traversable nodes have trav <= this (or gateway != 0)",
    )
    parser.add_argument(
        "--slope-max",
        type=float,
        default=0.2,
        help="max |dz/dxy| slope for a cell to be considered flat (ratio, tan theta). "
             "Slope is the boundary between layers.",
    )
    parser.add_argument(
        "--min-nodes",
        type=int,
        default=2000,
        help="minimum component size in nodes to keep",
    )
    parser.add_argument(
        "--dedup-iou",
        type=float,
        default=0.0,
        help="3D bbox IoU threshold for removing overlapping smaller regions. "
             "Default 0 means any overlap is forbidden. Set to -1 to disable deduplication.",
    )
    parser.add_argument(
        "--merge-z-gap",
        type=float,
        default=0.5,
        help="after deduplication, merge regions whose z-direction nearest distance "
             "is smaller than this value (meters). Set to -1 to disable merging.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = resolve_input(args.pickle)
    output_path = resolve_output(args.output, input_path)
    if output_path.is_dir():
        raise ValueError(f"--output must be a file path, not a directory: {output_path}")

    log(f"loading {input_path}")
    data = load_sparse_pickle(input_path)

    result = split_nav_map(
        data,
        cost_threshold=args.cost_threshold,
        slope_max=args.slope_max,
        min_nodes=args.min_nodes,
        dedup_iou=args.dedup_iou if args.dedup_iou >= 0.0 else None,
        merge_z_gap=args.merge_z_gap if args.merge_z_gap >= 0.0 else None,
    )

    result["source"] = str(input_path)
    result["params"] = {
        "cost_threshold": args.cost_threshold,
        "slope_max": args.slope_max,
        "min_nodes": args.min_nodes,
        "dedup_iou": args.dedup_iou if args.dedup_iou >= 0.0 else None,
        "merge_z_gap": args.merge_z_gap if args.merge_z_gap >= 0.0 else None,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    log(f"wrote {output_path} ({len(result['regions'])} regions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

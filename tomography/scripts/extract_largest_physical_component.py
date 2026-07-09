#!/usr/bin/env python3
"""Extract the largest planner-reachable component from a sparse tomogram pickle.

Connectivity is defined exactly by the Sparse A* planner's edge rules:

1. A node is traversable if trav <= cost_threshold OR gateway != 0.
2. Same-layer edges: 8-neighborhood, both nodes traversable,
   |elev_g difference| <= step_max.
3. Cross-layer edges: a gateway node (gateway > 0 up, < 0 down) connects to one
   target node in the adjacent layer within a 3x3 window (including itself),
   chosen by minimum Chebyshev distance then minimum cost, with the same
   traversability and height-difference checks.

The largest connected component under these rules is kept; everything else is
removed from the sparse pickle.
"""

import argparse
import os
import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph


RSG_ROOT = Path(__file__).resolve().parents[2]


def log(message: str) -> None:
    print(f"[extract-planner] {message}", flush=True)


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
        return input_path.with_name(f"{input_path.stem}_planner{input_path.suffix}")
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


def overlap_slices(dim: int, d: int):
    if d >= 0:
        return slice(0, dim - d), slice(d, dim)
    return slice(-d, dim), slice(0, dim + d)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep only the largest component reachable by the Sparse A* planner."
    )
    parser.add_argument("--pickle", type=str, default="scene_map_sparse.pickle", help="input sparse tomogram pickle")
    parser.add_argument("--output", type=str, default=None, help="output pickle; defaults to <input>_planner.pickle")
    parser.add_argument("--cost-threshold", type=float, default=35.0, help="traversable nodes have trav <= this")
    parser.add_argument("--step-max", type=float, default=0.5, help="max |elev_g| difference for any edge")
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
    elev_c = np.asarray(data_dict["elev_c"], dtype=np.float32)
    gateway = np.asarray(data_dict["gateway"], dtype=np.int32)
    shape = tuple(int(x) for x in data_dict["shape"])  # [S, Y, X]
    n_slice, n_y, n_x = shape
    n_nodes = indices.shape[0]

    log(f"nodes={n_nodes:,}, shape={shape}")

    # Dense helper arrays.
    node_id = np.full(shape, -1, dtype=np.int32)
    active = np.zeros(shape, dtype=bool)
    height = np.full(shape, np.nan, dtype=np.float32)
    cost = np.full(shape, np.nan, dtype=np.float32)
    gateway_dense = np.zeros(shape, dtype=np.int32)

    traversable = (trav <= float(args.cost_threshold)) | (gateway != 0)
    node_id[indices[:, 0], indices[:, 1], indices[:, 2]] = np.arange(n_nodes, dtype=np.int32)
    active[indices[:, 0], indices[:, 1], indices[:, 2]] = traversable
    height[indices[:, 0], indices[:, 1], indices[:, 2]] = elev_g
    cost[indices[:, 0], indices[:, 1], indices[:, 2]] = trav
    gateway_dense[indices[:, 0], indices[:, 1], indices[:, 2]] = gateway

    active_count = int(np.count_nonzero(active))
    log(f"active (traversable/gateway) nodes={active_count:,}")
    if active_count == 0:
        raise RuntimeError("no active nodes")

    # Collect same-layer edges (undirected: use only 4 unique directions).
    edge_rows = []
    edge_cols = []
    step_max = float(args.step_max)
    same_layer_offsets = [(0, 1), (1, -1), (1, 0), (1, 1)]

    for dr, dc in same_layer_offsets:
        row_base, row_nb = overlap_slices(n_y, dr)
        col_base, col_nb = overlap_slices(n_x, dc)
        bid = node_id[:, row_base, col_base]
        nid = node_id[:, row_nb, col_nb]
        b_active = active[:, row_base, col_base]
        n_active = active[:, row_nb, col_nb]
        b_height = height[:, row_base, col_base]
        n_height = height[:, row_nb, col_nb]

        valid = (
            (bid >= 0)
            & (nid >= 0)
            & b_active
            & n_active
            & (np.abs(b_height - n_height) <= step_max)
        )
        if np.any(valid):
            edge_rows.append(bid[valid])
            edge_cols.append(nid[valid])
        log(f"same-layer offset ({dr:2d},{dc:2d}): edges={int(np.count_nonzero(valid)):,}")

    # Cross-layer edges: one directed edge from each gateway node to its target.
    gateway_idx = np.flatnonzero(gateway != 0)
    log(f"gateway nodes={int(gateway_idx.size):,}")
    cross_rows = []
    cross_cols = []

    for i in gateway_idx:
        l = int(indices[i, 0])
        r = int(indices[i, 1])
        c = int(indices[i, 2])
        tl = l + 1 if gateway[i] > 0 else l - 1
        if tl < 0 or tl >= n_slice:
            continue
        h0 = float(elev_g[i])
        best_id = -1
        best_dist = 3  # max possible within 3x3 is 2
        best_cost = float("inf")
        for dr in (-1, 0, 1):
            rr = r + dr
            if rr < 0 or rr >= n_y:
                continue
            for dc in (-1, 0, 1):
                cc = c + dc
                if cc < 0 or cc >= n_x:
                    continue
                if not active[tl, rr, cc]:
                    continue
                if abs(float(height[tl, rr, cc]) - h0) > step_max:
                    continue
                dist = dr * dr + dc * dc
                cid = int(node_id[tl, rr, cc])
                co = float(cost[tl, rr, cc])
                if dist < best_dist or (dist == best_dist and co < best_cost):
                    best_id = cid
                    best_dist = dist
                    best_cost = co
        if best_id >= 0:
            cross_rows.append(i)
            cross_cols.append(best_id)

    if cross_rows:
        edge_rows.append(np.asarray(cross_rows, dtype=np.int32))
        edge_cols.append(np.asarray(cross_cols, dtype=np.int32))
    log(f"cross-layer edges={len(cross_rows):,}")

    # Build sparse graph and find connected components.
    all_rows = np.concatenate(edge_rows).astype(np.int32)
    all_cols = np.concatenate(edge_cols).astype(np.int32)
    data = np.ones(all_rows.shape[0], dtype=bool)
    graph = sparse.coo_matrix(
        (data, (all_rows, all_cols)), shape=(n_nodes, n_nodes)
    ).tocsr()

    log(f"total edges={all_rows.shape[0]:,}; running connected_components ...")
    n_components, labels = csgraph.connected_components(graph, directed=False, return_labels=True)
    log(f"components={n_components:,}")

    component_sizes = np.bincount(labels)
    largest_label = int(np.argmax(component_sizes))
    largest_size = int(component_sizes[largest_label])
    log(f"largest component: nodes={largest_size:,}, ratio={largest_size / active_count:.4f}")

    keep = labels == largest_label
    removed = int(np.count_nonzero(~keep))
    log(f"removed nodes={removed:,}")

    out = dict(data_dict)
    out["indices"] = indices[keep]
    out["trav"] = trav[keep]
    out["elev_g"] = elev_g[keep]
    out["elev_c"] = elev_c[keep]
    out["gateway"] = gateway[keep]
    out["connectivity_source"] = {
        "input": str(input_path),
        "method": "planner_reachability",
        "cost_threshold": float(args.cost_threshold),
        "step_max": float(args.step_max),
        "total_nodes": n_nodes,
        "active_nodes": active_count,
        "largest_nodes": largest_size,
        "removed_nodes": removed,
        "components": int(n_components),
        "same_layer_edges": int(all_rows.shape[0] - len(cross_rows)),
        "cross_layer_edges": len(cross_rows),
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

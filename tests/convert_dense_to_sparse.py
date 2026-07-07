#!/usr/bin/env python3
"""Convert an existing dense tomogram pickle into the sparse v1 format."""

import os
import sys
import pickle
import time
import numpy as np

# Must match tomography/scripts/tomography.py
SPARSE_ASTAR_COST_THRESHOLD = 35.0
SPARSE_GATEWAY_COST_DELTA = 8.0
SPARSE_GATEWAY_HEIGHT_DELTA = 0.1


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


def convert_dense_to_sparse(dense_path, sparse_path):
    print(f"Loading dense pickle: {dense_path}")
    with open(dense_path, 'rb') as f:
        dense = pickle.load(f)

    tomogram = np.asarray(dense['data'], dtype=np.float32)
    resolution = float(dense['resolution'])
    center = np.asarray(dense['center'], dtype=np.float64)
    slice_h0 = float(dense.get('slice_h0', 0.0))
    slice_dh = float(dense['slice_dh'])
    slice_heights = np.asarray(dense['slice_heights'], dtype=np.float32)

    n_slice, map_dim_x, map_dim_y = tomogram.shape[1], tomogram.shape[2], tomogram.shape[3]

    layers_t = tomogram[0]
    layers_g = tomogram[3]
    layers_c = tomogram[4]

    gateway = compute_gateway(layers_t, layers_g)

    valid_height = np.isfinite(layers_g)
    valid_cost = np.isfinite(layers_t)
    traversable = layers_t <= SPARSE_ASTAR_COST_THRESHOLD
    gateway_keep = gateway != 0

    sparse_mask = (
        valid_height
        & valid_cost
        & traversable
    ) | gateway_keep

    coords = np.argwhere(sparse_mask)
    indices = np.stack([coords[:, 0], coords[:, 2], coords[:, 1]], axis=1).astype(np.int32)

    sparse = {
        'format': 'tomogram_sparse_v1',
        'shape': [int(n_slice), int(map_dim_y), int(map_dim_x)],
        'resolution': resolution,
        'center': center,
        'slice_h0': slice_h0,
        'slice_dh': slice_dh,
        'slice_heights': slice_heights,
        'indices': indices,
        'trav': layers_t[sparse_mask].astype(np.float32),
        'elev_g': layers_g[sparse_mask].astype(np.float32),
        'elev_c': layers_c[sparse_mask].astype(np.float32),
        'gateway': gateway[sparse_mask].astype(np.int32),
    }

    os.makedirs(os.path.dirname(sparse_path), exist_ok=True)
    t0 = time.time()
    with open(sparse_path, 'wb') as f:
        pickle.dump(sparse, f, protocol=pickle.HIGHEST_PROTOCOL)
    elapsed_ms = (time.time() - t0) * 1e3

    dense_cells = int(n_slice) * int(map_dim_y) * int(map_dim_x)
    sparse_nodes = int(indices.shape[0])
    sparse_ratio = sparse_nodes / dense_cells if dense_cells > 0 else 0.0

    print(f"Sparse pickle written: {sparse_path}")
    print(f"  dense_cells = {dense_cells}")
    print(f"  sparse_nodes = {sparse_nodes}")
    print(f"  sparse_ratio = {sparse_ratio:.4f}")
    print(f"  file_size = {os.path.getsize(sparse_path) / (1024.0 * 1024.0):.2f} MB")
    print(f"  write_time = {elapsed_ms:.2f} ms")


if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dense_path = os.path.join(root, 'rsc', 'tomogram', 'scene_map.pickle')
    sparse_path = os.path.join(root, 'rsc', 'tomogram', 'scene_map_sparse.pickle')
    convert_dense_to_sparse(dense_path, sparse_path)

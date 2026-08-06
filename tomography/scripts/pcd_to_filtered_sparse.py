#!/usr/bin/env python3
"""PCD → filtered sparse tomogram pickle.

Calls two existing scripts in sequence:
1. tomography/scripts/tomography.py   : PCD → scene_map_sparse.pickle
2. tomography/scripts/extract_largest_physical_component.py
                                       : keep largest planner-reachable component

The first script is a ROS node that keeps spinning after export; we terminate it
as soon as the sparse pickle has been written.
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


RSG_ROOT = Path(__file__).resolve().parents[2]
TOMOGRAPHY_DIR = RSG_ROOT / "tomography" / "scripts"
TOMOGRAM_DIR = RSG_ROOT / "rsc" / "tomogram"
PCD_DIR = RSG_ROOT / "rsc" / "pcd"

SPARSE_PICKLE = TOMOGRAM_DIR / "scene_map_sparse.pickle"
FILTERED_PICKLE = TOMOGRAM_DIR / "scene_map_sparse_planner.pickle"


def log(message: str) -> None:
    print(f"[pcd-to-filtered-sparse] {message}", flush=True)


def resolve_pcd(path: str) -> str:
    """Return the PCD file name as expected by tomography.py (under rsc/pcd/)."""
    p = Path(path).expanduser()
    if p.is_absolute():
        p = p.resolve()
        if p.parent.resolve() == PCD_DIR.resolve():
            return p.name
        # Copy into rsc/pcd so tomography.py can find it.
        dest = PCD_DIR / p.name
        if not dest.exists():
            log(f"copying {p} → {dest}")
            shutil.copy2(p, dest)
        return dest.name
    # Relative basename assumed under rsc/pcd.
    if (PCD_DIR / path).is_file():
        return path
    raise FileNotFoundError(f"PCD not found in {PCD_DIR}: {path}")


def resolve_output(path: str | None) -> Path:
    if not path:
        return FILTERED_PICKLE
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (TOMOGRAM_DIR / p).resolve()


def run_tomography(
    pcd_name: str,
    timeout: float = 600.0,
    tiled: bool = False,
    gpu_memory_gb: float = 10.0,
    tile_size: int | None = None,
    temp_dir: str | None = None,
) -> None:
    """Run the selected generator until the sparse pickle is exported."""
    if tiled:
        cmd = [
            sys.executable,
            "tiled_tomography.py",
            "--pcd",
            pcd_name,
            "--output",
            str(SPARSE_PICKLE),
            "--gpu-memory-gb",
            str(gpu_memory_gb),
        ]
        if tile_size is not None:
            cmd.extend(["--tile-size", str(tile_size)])
        if temp_dir is not None:
            cmd.extend(["--temp-dir", temp_dir])
    else:
        cmd = [
            "bash",
            "-c",
            f"source /opt/ros/humble/setup.bash && python3 tomography.py --pcd {pcd_name}",
        ]
    log(f"running: {' '.join(cmd)}  (cwd={TOMOGRAPHY_DIR})")

    proc = subprocess.Popen(
        cmd,
        cwd=TOMOGRAPHY_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )

    exported = False
    t0 = time.time()
    try:
        while True:
            line = proc.stdout.readline()
            if not line:
                # Process ended on its own.
                break
            line = line.rstrip()
            print(line, flush=True)
            if "Sparse tomogram exported:" in line:
                exported = True
                if tiled:
                    log("sparse pickle exported")
                else:
                    log("sparse pickle exported; terminating tomography node")
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                    break
            if time.time() - t0 > timeout:
                raise TimeoutError("tomography.py did not export sparse pickle in time")
    finally:
        try:
            proc.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            log("tomography node did not exit cleanly, sending SIGKILL")
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=10.0)

    if not exported:
        raise RuntimeError("tomography.py finished but sparse pickle was not exported")

    if not SPARSE_PICKLE.is_file():
        raise RuntimeError(f"expected sparse pickle missing: {SPARSE_PICKLE}")


def run_filter(cost_threshold: float, step_max: float) -> None:
    cmd = [
        "python3",
        "extract_largest_physical_component.py",
        "--pickle", str(SPARSE_PICKLE),
        "--output", str(FILTERED_PICKLE),
        "--cost-threshold", str(cost_threshold),
        "--step-max", str(step_max),
    ]
    log(f"running: {' '.join(cmd)}  (cwd={TOMOGRAPHY_DIR})")
    subprocess.run(cmd, cwd=TOMOGRAPHY_DIR, check=True)
    if not FILTERED_PICKLE.is_file():
        raise RuntimeError(f"expected filtered pickle missing: {FILTERED_PICKLE}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a sparse tomogram from a PCD and keep only the largest planner-reachable component."
    )
    parser.add_argument("--pcd", type=str, required=True, help="input PCD file (absolute path or basename under rsc/pcd/)")
    parser.add_argument("--output", type=str, default=None, help="output pickle path; defaults to rsc/tomogram/scene_map_sparse_planner.pickle")
    parser.add_argument("--cost-threshold", type=float, default=35.0, help="passed to the connectivity filter")
    parser.add_argument("--step-max", type=float, default=0.5, help="passed to the connectivity filter")
    parser.add_argument("--timeout", type=float, default=600.0, help="max seconds to wait for tomography export")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--tiled",
        dest="tiled",
        action="store_true",
        help="use the exact memory-bounded tiled implementation (default)",
    )
    mode.add_argument(
        "--legacy",
        dest="tiled",
        action="store_false",
        help="use the original single-allocation tomography.py path",
    )
    parser.set_defaults(tiled=True)
    parser.add_argument("--gpu-memory-gb", type=float, default=10.0, help="GPU memory budget for --tiled")
    parser.add_argument("--tile-size", type=int, default=None, help="force square core tiles for equivalence testing")
    parser.add_argument("--temp-dir", type=str, default=None, help="temporary memmap directory for --tiled")
    parser.add_argument("--remove-intermediate", action="store_true", help="remove scene_map_sparse.pickle after copying output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = resolve_output(args.output)

    pcd_name = resolve_pcd(args.pcd)
    log(f"input PCD file_name={pcd_name}")
    log(f"output pickle={output_path}")

    run_tomography(
        pcd_name,
        timeout=args.timeout,
        tiled=args.tiled,
        gpu_memory_gb=args.gpu_memory_gb,
        tile_size=args.tile_size,
        temp_dir=args.temp_dir,
    )
    run_filter(args.cost_threshold, args.step_max)

    if output_path != FILTERED_PICKLE:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        log(f"copying filtered pickle → {output_path}")
        shutil.copy2(FILTERED_PICKLE, output_path)

    if args.remove_intermediate and output_path != SPARSE_PICKLE:
        log(f"removing intermediate {SPARSE_PICKLE}")
        try:
            SPARSE_PICKLE.unlink()
        except FileNotFoundError:
            pass

    log(f"done: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

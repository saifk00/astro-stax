"""Measure warm flat stacking on a reproducible sequence of synthetic uint16 FITS files."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
import triton
from astropy.io import fits

from stax import Sequence


def make_flats(directory: Path, frames: int, height: int, width: int) -> list[Path]:
    rng = np.random.default_rng(5070)
    base = 3000 + np.linspace(0, 250, width, dtype=np.int16)
    paths = []
    for i in range(frames):
        pixels = rng.integers(-80, 81, size=(height, width), dtype=np.int16)
        pixels += base + (i % 13) * 35
        pixels[i % 19::19, i % 23::23] = 12000
        pixels[::97, ::83] = 0
        path = directory / f"flat_{i + 1:05d}.fits"
        fits.writeto(path, pixels.astype(np.uint16))
        paths.append(path)
    return paths


def measure(paths: list[Path], output: Path) -> dict:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    frames, levels = Sequence.from_fits(paths, return_levels=True)
    torch.cuda.synchronize()
    ready = perf_counter()
    master = frames.stack_flats(levels_adu=levels)
    torch.cuda.synchronize()
    stacked = perf_counter()
    master.save(output)
    return dict(resident_s=stacked - ready, end_to_end_s=perf_counter() - start,
                peak_tensor_bytes=torch.cuda.max_memory_allocated())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=int, nargs="+", default=[8, 21, 64, 128])
    parser.add_argument("--shape", type=int, nargs=2, default=[4000, 6000], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.shape) < 1 or not all(2 <= n <= 1024 for n in args.counts) or args.repeats < 1:
        parser.error("Use positive dimensions and repeats, and frame counts between 2 and 1024.")
    props = torch.cuda.get_device_properties(0)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                                     text=True).strip()
    smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    driver = None
    if Path(smi).is_file():
        query = subprocess.run([smi, "--query-gpu=driver_version", "--format=csv,noheader"],
                               capture_output=True, text=True)
        if query.returncode == 0 and query.stdout.strip():
            driver = query.stdout.splitlines()[0]
    cpu = next((line.partition(":")[2].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), platform.machine())
    result = dict(commit=commit, driver=driver, cpu=cpu, gpu=props.name, vram_bytes=props.total_memory,
                  compute_capability=[props.major, props.minor], platform=platform.platform(),
                  cpu_count=os.cpu_count(), torch=str(torch.__version__), cuda=torch.version.cuda,
                  triton=triton.__version__, shape=args.shape, seed=5070, cases=[])
    with TemporaryDirectory(dir=Path.cwd()) as tmp:
        directory = Path(tmp)
        paths = make_flats(directory, max(args.counts), *args.shape)
        master = directory / "master.fits"
        for n in args.counts:
            torch.cuda.empty_cache()
            try:
                measure(paths[:n], master)  # Prime filesystem, allocator, and Triton caches.
                runs = [measure(paths[:n], master) for _ in range(args.repeats)]
            except torch.OutOfMemoryError:
                case = dict(frames=n, status="out_of_memory")
            else:
                with fits.open(master, memmap=True) as hdus:
                    digest = hashlib.sha256(memoryview(hdus[0].data).cast("B")).hexdigest()
                case = dict(frames=n, status="ok", resident_s=median(r["resident_s"] for r in runs),
                            end_to_end_s=median(r["end_to_end_s"] for r in runs),
                            peak_tensor_bytes=max(r["peak_tensor_bytes"] for r in runs),
                            master_sha256=digest, runs=runs)
            result["cases"].append(case)
            print(json.dumps({k: v for k, v in case.items() if k != "runs"}), flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not any(c["status"] == "ok" for c in result["cases"]):
        raise SystemExit("No requested stack fit in GPU memory.")


if __name__ == "__main__":
    main()

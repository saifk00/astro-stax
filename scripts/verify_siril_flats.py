"""Time flat stacking from input FITS to a host master, then compare its pixels with Siril."""

from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from stax import Image, Sequence
from stax.io import _load_u16_cuda, read_fits
from stax._siril_compat import stack_siril_u16

ROOT = Path(__file__).resolve().parents[1]


def stack_once(files, reference, host_levels, *, pin_memory, packed_warp=False):
    # Initialize CUDA and the activity collector before the first timed stage.
    # The stack wall time includes first-call dispatch and any Triton compilation.
    torch.cuda.synchronize()
    activities = [torch.profiler.ProfilerActivity.CPU,
                  {torch.profiler.ProfilerActivity.CUDA: ["CONCURRENT_KERNEL"]}]
    with torch.profiler.profile(activities=activities) as prof:
        start = perf_counter()
        with torch.profiler.record_function("stax.fits.upload"):
            levels = None if host_levels is None else host_levels.to("cuda")
        if packed_warp:
            frames, _, measured = _load_u16_cuda(files, "cuda", pixel_major=True,
                                                  return_levels=host_levels is None)
            if host_levels is None:
                levels = measured
        elif host_levels is None:
            flats, levels = Sequence.from_fits(files, pin_memory=pin_memory, return_levels=True)
        else:
            flats = Sequence.from_fits(files, pin_memory=pin_memory)
        torch.cuda.synchronize()
        ready = perf_counter()

        with torch.profiler.record_function("stax.verify.stack"):
            if packed_warp:
                result = stack_siril_u16(frames.flatten(1), levels, reference=reference,
                                         normalization="mul", rejection="winsorized",
                                         one_warp_per_pixel=True)
                master = Image.wrap(result.reshape(frames.shape[1:]))
            else:
                master = flats.stack_flats(levels_adu=levels, reference=reference)
            torch.cuda.synchronize()
        stacked = perf_counter()

        actual = master.to_host(pin_memory=pin_memory)
        downloaded = perf_counter()

    events = prof.events()
    read_s = sum(e.cpu_time_total for e in events if e.name == "stax.fits.read") / 1e6
    upload_s = sum(e.cpu_time_total for e in events if e.name == "stax.fits.upload") / 1e6
    kernels = [e for e in events if e.device_type == torch.autograd.DeviceType.CUDA]
    decode = [e for e in kernels if e.name in ("_decode_fits_u16", "_decode_packed_fits")]
    stack_range = next(e.time_range for e in events if e.name == "stax.verify.stack")
    stack = [e for e in kernels if stack_range.start <= e.time_range.start <= stack_range.end]

    print("Host buffers: one pinned input frame" if pin_memory else "Host buffers: pageable batch")
    print(f"Input FITS to ready GPU sequence + levels {ready - start:9.4f} s")
    print(f"  FITS read{' and CPU decode' if not pin_memory else ''}: {read_s:.4f} s")
    print(f"  Host to device upload: {upload_s:.4f} s")
    if decode:
        label = "decode + histogram" if host_levels is None else "decode"
        print(f"  GPU FITS {label} kernel: {sum(e.device_time_total for e in decode) / 1e6:.4f} s")
    print(f"Resident stack (wall)                     {stacked - ready:9.4f} s")
    if stack:
        kernel_seconds = sum(e.device_time_total for e in stack) / 1e6
        print(f"  GPU stack kernels ({len(stack)} launches): {kernel_seconds:.4f} s")
    else:
        print("  GPU stack kernel time unavailable")
    print(f"Device to host download (incl. allocation) {downloaded - stacked:9.4f} s")
    print(f"End to end (input FITS to host master)    {downloaded - start:9.4f} s")
    return actual


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=ROOT / "test_data/fits/flats")
    parser.add_argument("--fresh-levels", action="store_true",
                        help="Compare a master made with freshly measured normalization statistics")
    parser.add_argument("--pageable", action="store_true",
                        help="Use pageable host memory for both transfers (default: pinned)")
    parser.add_argument("--packed-warp", action="store_true",
                        help="Decode into packed pixel stacks and assign one warp to each pixel (2–32 flats)")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Repeat the full timed pipeline in one process to measure buffer reuse")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.packed_warp and args.pageable:
        parser.error("--packed-warp uses the pinned raw FITS decoder")
    # This fixture is a single-layer flat_ sequence with every frame included.
    records = [line.split() for line in (args.directory / "flat_.seq").read_text().splitlines()]
    header = next(row for row in records if row[:1] == ["S"])
    indices = [int(row[1]) for row in records if row[:1] == ["I"] and row[2] == "1"]
    if len(indices) != int(header[3]):
        parser.error("Expected a flat_ sequence with all frames selected")
    reference = max(0, int(header[6]))  # Unregistered sequence defaults to first included frame.
    files = [args.directory / f"flat_{i:0{int(header[5])}d}.fits" for i in indices]
    host_levels = None
    if not args.fresh_levels:
        # M<layer>-<frame> statistics field 9 is the location in ADU.
        locations = {int(row[0][3:]): float(row[9]) for row in records if row and row[0].startswith("M0-")}
        host_levels = torch.tensor([locations[i] for i in range(len(files))], dtype=torch.float64)

    expected = None
    for run in range(args.repeats):
        if args.repeats > 1:
            print(f"Run {run + 1} of {args.repeats}", flush=True)
        actual = stack_once(files, reference, host_levels, pin_memory=not args.pageable,
                            packed_warp=args.packed_warp)
        if expected is None:
            expected, _ = read_fits(args.directory / "master_flat.fits")
        if expected.shape != actual.shape or expected.dtype != np.float32:
            parser.error("Expected a float32 master with the input frame geometry")
        different = np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32))
        print(f"{different:,} of {expected.size:,} float32 pixels differ")
        if different:
            return 1
        del actual  # Release the pinned output before allocating the next one.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

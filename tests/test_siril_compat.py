"""Bitwise GPU checks against actual Siril code, including guard-limited pixels."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import torch
import triton
import triton.language as tl

from stax import Sequence
from stax._background import INV_USHORT, estimate_backgrounds_u16
from stax.io import read_fits
from stax._siril_compat import _fast, _small_stack_variance, _variance_reciprocals, stack_siril_u16

ROOT = Path(__file__).resolve().parents[1]


@triton.jit
def _variance_probe(sums, divisors, output, reciprocals):
    i = tl.program_id(0)
    tl.store(output + i, _small_stack_variance(tl.load(sums + i), tl.load(divisors + i), reciprocals))


class CudaCompilationTests(unittest.TestCase):
    def test_compile_winsorized_flat_kernel_for_rtx_5070_ti(self):
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource

        # Compile the 21-flat specialization without initializing a CUDA device.
        signature = dict(source="*u16", coefficients="*fp64", output="*fp32", counts="*i32",
                         steps_out="*i16", replay_out="*i1", pixels="i32", reciprocals="*fp64")
        constants = dict(N=21, SIGMA=3.0, CAP=21, BP=1, BF=32, MULTIPLY=True, WINSOR=True,
                         PACKED=False, refinements_out=None, PACK_REJECTION_COUNTS=False,
                         survivor_bits_out=None)
        signature.update(dict.fromkeys(constants, "constexpr"))
        kernel = triton.compile(ASTSource(_fast, signature, constants),
                                target=GPUTarget("cuda", 120, 32),
                                options={"num_warps": 1, "enable_fp_fusion": False})
        self.assertTrue(kernel.asm["cubin"])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SmallStackVarianceTests(unittest.TestCase):
    def test_final_mean_with_zero_slots_and_large_sums(self):
        rng = np.random.default_rng(65535)
        for n in (32, 256, 513):
            # Near-saturation inputs exercise all survivor counts, including large
            # sums that would lose low bits if the tiled path switched to FP32.
            raw = rng.integers(65000, 65536, size=(n, n + 1), dtype=np.uint16)
            counts = np.arange(n + 1)
            raw[np.arange(n)[:, None] >= counts] = 0
            mean = (raw.sum(axis=0, dtype=np.int64) / np.maximum(counts, 1)).astype(np.float32)
            expected = mean * np.float32(INV_USHORT)
            out = stack_siril_u16(torch.from_numpy(raw).cuda(), sigma=1000., normalization="none")
            np.testing.assert_array_equal(out.cpu().numpy().view(np.uint32), expected.view(np.uint32))

    def test_rounding_midpoints(self):
        # Include both even and odd FP32 significands and exponent boundaries.
        bases = np.array([0.03125, 1, 127.99999, 4096, 1e8, 2**30], dtype=np.float32)
        lower = np.stack([np.nextafter(bases, np.float32(-np.inf)), bases,
                          np.nextafter(bases, np.float32(np.inf))]).ravel()
        upper = np.nextafter(lower, np.float32(np.inf))
        midpoints = (lower.astype(np.float64) + upper.astype(np.float64)) / 2
        divisors = np.arange(1, 256, dtype=np.float64)
        exact = midpoints[:, None] * divisors
        # Probe both sides of each tie, including values outside the guard.
        offsets = np.arange(-8, 9, dtype=np.int64)
        sums = (exact.view(np.int64)[:, :, None] + offsets).view(np.float64)
        divisors = np.broadcast_to(divisors[None, :, None], sums.shape).flatten()
        sums = sums.flatten()
        expected = (sums / divisors).astype(np.float32)
        unguarded = (sums * (1.0 / divisors)).astype(np.float32)
        self.assertTrue(np.any(unguarded != expected))
        output = torch.empty(expected.size, dtype=torch.float32, device="cuda")
        _variance_probe[(expected.size,)](torch.from_numpy(sums).cuda(),
                                          torch.from_numpy(divisors).cuda(), output,
                                          _variance_reciprocals(output.device))
        np.testing.assert_array_equal(output.cpu().numpy().view(np.uint32), expected.view(np.uint32))


def write_fits(path, values):
    bitpix = 16 if values.dtype == np.uint16 else -32
    height, width = values.shape
    cards = ["SIMPLE  =                    T", f"BITPIX  = {bitpix:20d}",
             "NAXIS   =                    2", f"NAXIS1  = {width:20d}", f"NAXIS2  = {height:20d}"]
    if bitpix == 16:
        cards += ["BZERO   =                32768", "BSCALE  =                    1"]
        payload = (values.astype(np.int32) - 32768).astype(">i2").tobytes()
    else:
        payload = values.astype(">f4").tobytes()
    cards.append("END")
    header = "".join(card.ljust(80) for card in cards)
    header = header.ljust((len(header) + 2879) // 2880 * 2880)
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(payload)
        stream.write(bytes(-len(payload) % 2880))


@unittest.skipUnless(torch.cuda.is_available() and (ROOT / ".local/siril-reference").exists(),
                     "Built Siril reference and CUDA required")
class SirilCompatibilityTests(unittest.TestCase):
    def test_bitwise_against_siril_with_tight_sigmas_and_zero_pixels(self):
        rng = np.random.default_rng(17)
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as temp:
            directory = Path(temp)
            for n in (3, 7, 21):
                raw = (rng.normal(2500, 100, (n, 2048)) + np.arange(n)[:, None] * 4).round().astype(np.uint16)
                raw[:, ::53] = 0
                raw[:, -1] = 65000  # Exercise the estimator's +/-6 MAD trim.
                files = []
                for i, frame in enumerate(raw):
                    path = directory / f"f{i}.fits"
                    write_fits(path, frame.reshape(1, -1))
                    files.append(path)
                manifest = directory / "inputs.txt"
                manifest.write_text("".join(str(p) + "\n" for p in files))
                frames = torch.from_numpy(raw).cuda()
                levels = estimate_backgrounds_u16(frames)
                for sigma in (1.25, 2.0, 3.0):
                    with self.subTest(n=n, sigma=sigma):
                        prefix = directory / "reference"
                        process = subprocess.run([str(ROOT / ".local/siril-reference"), str(manifest),
                                                  "2048", "1", "4", str(sigma), "1", str(prefix)],
                                                 text=True, capture_output=True, check=True)
                        expected_levels = np.array([float(line.split()[2]) for line in process.stdout.splitlines()
                                                    if line.startswith("BACKGROUND")])
                        expected = np.fromfile(str(prefix) + ".f32", dtype=np.float32)
                        output, profile = stack_siril_u16(frames, levels, sigma=sigma, return_profile=True)
                        np.testing.assert_array_equal(
                            torch.floor(levels[0] - levels + 0.5).cpu().numpy(),
                            np.floor(expected_levels[0] - expected_levels + 0.5))
                        np.testing.assert_array_equal(output.cpu().numpy().view(np.uint32), expected.view(np.uint32))
                        if n == 21 and sigma == 1.25:
                            self.assertGreater(profile["replayed_pixels"], 0)


@unittest.skipUnless(torch.cuda.is_available() and (ROOT / ".local/siril/bin/siril-cli").exists(),
                     "Built Siril CLI and CUDA required")
class RawFlatStackTests(unittest.TestCase):
    def test_raw_master_flat_against_script(self):
        for n in (21, 64, 129, 256):
            with self.subTest(n=n):
                self.check_flat_master(n)

    def check_flat_master(self, n):
        rng = np.random.default_rng(95)
        raw = rng.normal(3000, 50, (n, 31, 33)) * np.linspace(0.8, 1.2, n)[:, None, None]
        raw = raw.round().astype(np.uint16)
        raw[:, 0, 0] = 0
        raw[1:, 0, 1] = 0
        raw[-1, 2] = 65535
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as tmp:
            directory = Path(tmp)
            for i, plane in enumerate(raw, start=1):
                write_fits(directory / f"flat_{i:05d}.fit", plane)
            env = os.environ | {"XDG_CONFIG_HOME": str(directory / "config"),
                                "XDG_DATA_HOME": str(ROOT / ".cache/siril-runtime/data"),
                                "PIP_NO_INDEX": "1"}
            subprocess.run([str(ROOT / ".local/siril/bin/siril-cli"), "--offline",
                            "-d", tmp, "-s", str(ROOT / "scripts/stack_siril_flats.ssf")],
                           env=env, check=True, capture_output=True, timeout=120)
            expected, _ = read_fits(directory / "master_flat.fits")
            frames = Sequence.upload(raw)
            for warp in (False, True):
                actual = frames.stack_flats(one_warp_per_pixel=warp).to_host()
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


if __name__ == "__main__":
    unittest.main()

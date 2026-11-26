"""Storage ownership, resident chaining, FITS units and PSF numerical accuracy."""

import gc
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from astropy.io import fits

from stax import Image, Sequence
from stax._background import INV_USHORT, _levels_from_histogram, estimate_backgrounds_u16
from stax.io import _load_u16_cuda, load_fits_host, read_fits, write_fits
from stax._siril_compat import _fast, stack_siril_u16


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ImageTests(unittest.TestCase):
    def test_sequence_from_resident_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f"flat{i}.fits" for i in range(3)]
            for i, path in enumerate(paths):
                write_fits(path, np.full((3, 5), 300 + i, dtype=np.uint16))
            images = [Image.from_fits(path) for path in paths]
            assembled = Sequence.from_images(reversed(images))
            loaded = Sequence.from_fits(reversed(paths))
            self.assertEqual(len(assembled), 3)
            self.assertEqual(assembled.metadata["paths"], loaded.metadata["paths"])
            self.assertIsInstance(assembled[:2], Sequence)
            self.assertIsInstance(assembled[0], Image)
            self.assertEqual(assembled[:2][-1].metadata, images[1].metadata)
            self.assertEqual(assembled[0].tensor.data_ptr(), assembled.tensor.data_ptr())
            torch.testing.assert_close(assembled.tensor, loaded.tensor)
            torch.testing.assert_close(assembled.stack_flats().tensor, loaded.stack_flats().tensor)

    def test_image_and_sequence_inputs(self):
        plane = torch.zeros((3, 4), device="cuda")
        with self.assertRaises(ValueError):
            Image.wrap(plane[None])
        with self.assertRaises(ValueError):
            Sequence.wrap(plane)
        with self.assertRaises(TypeError):
            Image.from_fits(["a.fits"])
        with self.assertRaises(TypeError):
            Sequence.from_fits("a.fits")
        with self.assertRaises(ValueError):
            Sequence.from_images([])
        with self.assertRaises(ValueError):
            Sequence.from_images([Image.wrap(plane), Image.wrap(plane[:1])])
        with self.assertRaises(ValueError):
            Sequence.from_images([Image.wrap(plane), Image.wrap(plane.to(torch.uint16))])

    def test_default_flat_stack_warp_boundary(self):
        for n, warps in ((32, 1), (33, 1), (256, 1), (257, 4)):
            with self.subTest(frames=n):
                frames = Sequence.upload(np.full((n, 3, 5), 4000, dtype=np.uint16))
                levels = torch.full((n,), 4000.0, dtype=torch.float64, device=frames.device)
                with patch.object(_fast, "run", wraps=_fast.run) as launch:
                    automatic = frames.stack_flats(levels_adu=levels)
                self.assertEqual(launch.call_args.kwargs["num_warps"], warps)
                tiled = frames.stack_flats(levels_adu=levels, one_warp_per_pixel=False)
                torch.testing.assert_close(automatic.tensor, tiled.tensor, rtol=0, atol=0)

    def test_packed_fits_tail_and_rejection_replay(self):
        rng = np.random.default_rng(21)
        for n in (7, 32):
            planes = rng.integers(0, 65536, (n, 5, 13), dtype=np.uint16)
            planes[:, 0, 0] = 0
            with tempfile.TemporaryDirectory() as tmp:
                files = [Path(tmp) / f"flat{i}.fits" for i in range(n)]
                for path, plane in zip(files, planes):
                    write_fits(path, plane)
                frames, _, levels = _load_u16_cuda(files, "cuda", return_levels=True)
                packed, _, packed_levels = _load_u16_cuda(files, "cuda", pixel_major=True, return_levels=True)
                self.assertEqual(packed.flatten(1).stride(), (1, n))
                np.testing.assert_array_equal(packed.cpu().numpy(), planes)
                torch.testing.assert_close(packed_levels, levels, rtol=0, atol=0)
                options = dict(normalization="mul", rejection="winsorized", sigma=0.2, return_profile=True)
                base, profile = stack_siril_u16(frames.flatten(1), levels, one_warp_per_pixel=False, **options)
                self.assertGreater(profile["replayed_pixels"], 0)
                actual, observed = stack_siril_u16(packed.flatten(1), packed_levels,
                                                   one_warp_per_pixel=True, **options)
                torch.testing.assert_close(actual, base, rtol=0, atol=0)
                torch.testing.assert_close(observed["survivors"], profile["survivors"])
                self.assertEqual(observed["replayed_pixels"], profile["replayed_pixels"])

    def test_fused_fits_histogram_and_levels(self):
        ramp = np.arange(75 * 61, dtype=np.uint16).reshape(75, 61)
        planes = [ramp, ~ramp, np.full_like(ramp, 32768), np.zeros_like(ramp)]
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f"flat{i}.fits" for i in range(len(planes))]
            for path, plane in zip(paths, planes):
                write_fits(path, plane)
            with patch("stax.io._levels_from_histogram", wraps=_levels_from_histogram) as reduce:
                image, levels = Sequence.from_fits(paths, return_levels=True)
            histogram, = reduce.call_args.args
            for bins, plane in zip(histogram.cpu().numpy(), planes):
                np.testing.assert_array_equal(bins, np.bincount(plane[plane != 0], minlength=65536))
            np.testing.assert_array_equal(image.to_host(), planes)
            expected = estimate_backgrounds_u16(image.tensor.flatten(1))
            torch.testing.assert_close(levels, expected, rtol=0, atol=0, equal_nan=True)
            self.assertTrue(levels[-1].isnan().item())
            single, single_level = Image.from_fits(paths[0], return_levels=True, pin_memory=False)
            self.assertEqual(single.tensor.ndim, 2)
            torch.testing.assert_close(single_level, levels[:1], rtol=0, atol=0)

    def test_single_buffer_fits_decode(self):
        raw = np.arange(21 * 33, dtype=np.uint16).reshape(21, 33)
        raw[0, :6] = [0, 255, 256, 32767, 32768, 65535]
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.fits", Path(tmp) / "b.fits"
            write_fits(a, raw)
            write_fits(b, ~raw, header={f"KEY{i}": i for i in range(50)})
            preadv = os.preadv

            def short_read(fd, buffers, offset):
                return preadv(fd, [buffers[0][:101]], offset)

            with patch("stax.io.os.preadv", side_effect=short_read), \
                 patch("stax.image.load_fits_host", side_effect=AssertionError("full host batch")):
                batch = Sequence.from_fits(iter([a, b, a]))
            np.testing.assert_array_equal(batch.to_host(), np.stack([raw, ~raw, raw]))
            self.assertEqual(batch.metadata["fits_headers"][1]["KEY49"], 49)
            write_fits(b, raw.T)
            with self.assertRaisesRegex(ValueError, "same shape"):
                Sequence.from_fits([a, b])
            b.write_bytes(a.read_bytes()[:2880 + raw.nbytes - 1])
            with self.assertRaisesRegex(ValueError, "truncated"):
                Sequence.from_fits([a, b])

    def test_fits_cpu_decoder_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "float.fits"
            pixels = np.array([[-0., -123., 32767., np.nan]], dtype=np.float32)
            write_fits(path, pixels)
            for units in ("normalized", "adu"):
                batch = Sequence.from_fits(iter([path]), float_units=units)
                reference, _ = read_fits(path, float_units=units)
                np.testing.assert_array_equal(batch.to_host().view(np.uint32),
                                              reference[None].view(np.uint32))
            # Scaled signed int16 also decodes to normalized float32.
            write_fits(path, np.array([[0, 32768, 65535]], dtype=np.uint16))
            payload = path.read_bytes()
            path.write_bytes(payload.replace(b"BZERO   =                32768",
                                             b"BZERO   =                  100", 1))
            np.testing.assert_array_equal(Image.from_fits(path).to_host(), read_fits(path)[0])

    def test_wrapping_views_dlpack_and_ownership(self):
        tensor = torch.ones((3, 12, 16), device="cuda")
        pointer = tensor.data_ptr()
        image = Sequence.wrap(tensor)
        self.assertEqual(image.tensor.data_ptr(), pointer)
        self.assertEqual(image[1].tensor.data_ptr(), pointer + 12 * 16 * 4)
        shared = torch.from_dlpack(image)
        self.assertEqual(shared.data_ptr(), pointer)
        del tensor, shared
        gc.collect()
        self.assertEqual(image.tensor.sum().item(), 3 * 12 * 16)

    def test_upload_units_and_explicit_fits_roundtrip(self):
        raw = np.arange(8 * 9, dtype=np.uint16).reshape(8, 9) * 100
        metadata = {"BAYERPAT": "RGGB", "EXPTIME": 10., "OBJECT": "M31 / Andromeda", "OBSERVER": "O'Brien"}
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "raw.fits", Path(tmp) / "out.fits"
            write_fits(first, raw, header=metadata)
            image = Image.from_fits(first)
            np.testing.assert_array_equal(image.to_host(), raw)
            self.assertEqual(image.metadata["fits_headers"][0]["OBJECT"], "M31 / Andromeda")
            converted = Image.upload(raw.astype(np.float32), float_units="adu", metadata=image.metadata)
            converted.save(second)
            actual, header = read_fits(second)
            np.testing.assert_array_equal(actual, raw.astype(np.float32) * np.float32(INV_USHORT))
            self.assertEqual(header["OBSERVER"], "O'Brien")
            batch = Sequence.from_fits([first, first], workers=2)
            self.assertEqual(batch.shape, (2, 8, 9))
            self.assertEqual(batch[1].metadata["paths"], (str(first),))
            for pinned in (True, False):
                inputs, _ = load_fits_host([first, first], pin_memory=pinned)
                self.assertEqual(torch.from_numpy(inputs).is_pinned(), pinned)
                output = batch.to_host(pin_memory=pinned)
                self.assertEqual(torch.from_numpy(output).is_pinned(), pinned)
                np.testing.assert_array_equal(output, inputs)
            from_adu = Image.upload(raw.astype(np.float32), float_units="adu")
            np.testing.assert_array_equal(from_adu.to_host(), actual)
            # Negative zero survives float32 FITS round trips.
            actual[0, 0] = -0.
            write_fits(second, actual)
            self.assertTrue(np.signbit(read_fits(second)[0][0, 0]))

    def test_named_versioned_hdu_raw_upload_and_master_header(self):
        raw = np.array([[0, 1, 32768], [65535, 400, 800]], dtype=np.uint16)
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "multi.fits", Path(tmp) / "master.fits"
            first = fits.ImageHDU(~raw, name="SCI")
            second = fits.ImageHDU(raw, name="SCI")
            second.ver = 2
            second.header["BAYERPAT"] = "RGGB"
            second.header["EXPTIME"] = 2.5
            second.header.add_history("first entry")
            second.header.add_history("second entry")
            fits.HDUList([fits.PrimaryHDU(), first, second]).writeto(src)
            with patch("stax.image.load_fits_host", side_effect=AssertionError("host decode")):
                flats, levels = Sequence.from_fits([src, src], hdu=("SCI", 2), return_levels=True)
                np.testing.assert_array_equal(Image.from_fits(src, hdu="SCI").to_host(), ~raw)
            np.testing.assert_array_equal(flats[0].to_host(), raw)
            self.assertIsInstance(flats.metadata["fits_headers"][0], fits.Header)
            flats[0].save(dst)
            with fits.open(dst) as hdus:
                self.assertEqual(list(hdus[0].header["HISTORY"]), ["first entry", "second entry"])
                np.testing.assert_array_equal(hdus[0].data, raw)
            master = flats.stack_flats(levels_adu=levels)
            master.save(dst)
            with fits.open(dst) as hdus:
                self.assertEqual(hdus[0].header["NCOMBINE"], 2)
                self.assertEqual(hdus[0].header["BAYERPAT"], "RGGB")
                self.assertNotIn("EXPTIME", hdus[0].header)
                self.assertNotIn("BZERO", hdus[0].header)
                np.testing.assert_array_equal(hdus[0].data, master.to_host())

    def test_compressed_and_plain_hdus_share_one_sequence(self):
        raw = np.arange(24, dtype=np.uint16).reshape(4, 6)
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / name for name in ("plain.fits", "tile.fits", "outer.fits.gz")]
            for path, cls in zip(paths, (fits.ImageHDU, fits.CompImageHDU, fits.ImageHDU)):
                fits.HDUList([fits.PrimaryHDU(), cls(raw, name="SCI")]).writeto(path)
            for pinned in (True, False):
                seq = Sequence.from_fits(paths, hdu=1, pin_memory=pinned)
                np.testing.assert_array_equal(seq.to_host(), np.stack([raw] * len(paths)))
            np.testing.assert_array_equal(Image.from_fits(paths[-1], hdu="SCI").to_host(), raw)
            with self.assertRaises(KeyError):
                Image.from_fits(paths[0], hdu="MISSING")

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            Image.wrap(torch.zeros((3, 3)))
        with self.assertRaises(TypeError):
            Image.upload(np.ones((3, 3), dtype=np.float64))
        image = Image.wrap(torch.ones((3, 3), device="cuda"))
        self.assertFalse(hasattr(image, "stack"))
        for name in ("stack", "calibrate", "background", "measure_psf"):
            self.assertFalse(hasattr(Sequence, name))
            self.assertFalse(hasattr(Image, name))


if __name__ == "__main__":
    unittest.main()

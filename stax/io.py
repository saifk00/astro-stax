"""Astropy FITS I/O with a pinned raw-payload path for uncompressed uint16 images."""

import os
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch
import triton
import triton.language as tl
from astropy.io import fits
from astropy.io.fits import Header

from ._background import INV_USHORT, _levels_from_histogram

type FitsPath = str | os.PathLike[str]
type HDUSelector = int | str | tuple[str, int]


def _image_shape(image: fits.PrimaryHDU | fits.ImageHDU | fits.CompImageHDU,
                 path: FitsPath) -> tuple[int, int]:
    if (not isinstance(image, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU))
            or len(image.shape) != 2 or min(image.shape) < 1):
        raise ValueError(f"Expected a nonempty 2D FITS image HDU in {path}")
    if image.header["BITPIX"] not in (16, -32):
        raise ValueError("Supported FITS BITPIX values are 16 and -32")
    return tuple(image.shape)


def _image_layout(hdus: fits.HDUList, hdu: HDUSelector, stream: BinaryIO,
                  path: FitsPath) -> tuple[Header, tuple[int, int], int | None]:
    image = hdus[hdu]
    shape = _image_shape(image, path)
    header = image.header.copy()
    # Offsets in a compressed file refer to its decompressed stream. A tiled
    # image's on-disk data block contains a binary table rather than pixels.
    if isinstance(image, fits.CompImageHDU) or os.pread(stream.fileno(), 6, 0) != b"SIMPLE":
        return header, shape, None
    offset = hdus.fileinfo(hdu)["datLoc"]
    size = shape[0] * shape[1] * (abs(header["BITPIX"]) // 8)
    if os.fstat(stream.fileno()).st_size < offset + size:
        raise ValueError(f"Invalid or truncated FITS payload: {path}")
    return header, shape, offset


def _read_header(stream: BinaryIO, path: FitsPath) -> tuple[Header, tuple[int, int], int]:
    """Inspect the plain primary image used in the raw-offset benchmark scripts."""
    with os.fdopen(os.dup(stream.fileno()), "rb") as duplicate:
        duplicate.seek(0)
        with fits.open(duplicate, memmap=False, do_not_scale_image_data=True) as hdus:
            header, shape, offset = _image_layout(hdus, 0, stream, path)
    if offset is None:
        raise ValueError(f"The benchmark requires an uncompressed image: {path}")
    stream.seek(offset)
    return header, shape, offset


def _is_unsigned16(header: Header | Mapping[str, Any]) -> bool:
    return (header["BITPIX"] == 16 and float(header.get("BSCALE", 1)) == 1
            and float(header.get("BZERO", 0)) == 32768)


def read_fits(path: FitsPath, *, hdu: HDUSelector = 0,
              float_units: str = "normalized") -> tuple[np.ndarray, Header]:
    """Decode the selected HDU on the host in FITS row order and STAX units."""
    if float_units not in ("normalized", "adu"):
        raise ValueError("float_units must be 'normalized' or 'adu'")
    with fits.open(path, memmap=False, uint=True) as hdus:
        image = hdus[hdu]
        _image_shape(image, path)
        header = image.header.copy()  # Astropy may update scaling cards when accessing data.
        if _is_unsigned16(header):
            values = np.asarray(image.data, dtype=np.uint16)
        else:
            values = np.asarray(image.data, dtype=np.float32)
            if header["BITPIX"] == 16 or float_units == "adu":
                values = values * np.float32(INV_USHORT)
    return values, header


def _fits_paths(paths: FitsPath | Iterable[FitsPath], workers: int) -> tuple[list[Path], bool]:
    single = isinstance(paths, (str, os.PathLike))
    paths = [Path(paths)] if single else [Path(p) for p in paths]
    if not paths or not isinstance(workers, int) or workers < 1:
        raise ValueError("Supply at least one path and a positive worker count")
    return paths, single


def _fits_metadata(paths: Sequence[Path], headers: Sequence[Header]) -> dict[str, Any]:
    return {"paths": tuple(map(str, paths)), "fits_headers": tuple(headers),
            "row_order": "fits_bottom_to_top"}


@triton.jit
def _fits_adu(word):
    value = word.to(tl.uint32)
    return (((value << 8) | (value >> 8)) ^ 0x8000).to(tl.uint16)


@triton.jit
def _decode_fits_u16(words, count, BLOCK: tl.constexpr, histogram=None, FRAMES: tl.constexpr = 1):
    # Interleave frames so consecutive blocks update different histogram rows.
    # Decode-only launches treat the entire batch as one contiguous frame.
    frame = tl.program_id(0) % FRAMES
    tile = tl.program_id(0) // FRAMES
    i = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    address = words + frame.to(tl.int64) * count + i
    native = _fits_adu(tl.load(address, mask=i < count, other=0))
    tl.store(address, native, mask=i < count)
    if histogram is not None:
        bin_ptr = histogram + frame.to(tl.int64) * 65536 + native.to(tl.int32)
        tl.atomic_add(bin_ptr, 1, mask=(i < count) & (native != 0), sem="relaxed")


@triton.jit
def _decode_packed_fits(words, packed, histogram, pixels, N: tl.constexpr,
                        BF: tl.constexpr, BP: tl.constexpr):
    f = tl.arange(0, BF)
    p = tl.program_id(0) * BP + tl.arange(0, BP)
    valid = (f[:, None] < N) & (p[None, :] < pixels)
    raw = tl.load(words + f[:, None].to(tl.int64) * pixels + p[None, :], valid, other=0)
    adu = _fits_adu(raw)
    tl.store(packed + p[:, None].to(tl.int64) * N + f[None, :], tl.trans(adu), tl.trans(valid))
    if histogram is not None:
        bins = histogram + f[:, None] * 65536 + adu.to(tl.int32)
        tl.atomic_add(bins, 1, valid & (adu != 0), sem="relaxed")


def _read_payload(fd: int, offset: int, dest: memoryview) -> None:
    done = 0
    while done < len(dest):
        size = os.preadv(fd, [dest[done:]], offset + done)
        if size == 0:
            raise EOFError("Truncated FITS payload")
        done += size


def _load_u16_cuda(
    paths: Sequence[Path], device: str | torch.device, *, return_levels: bool = False,
    pixel_major: bool = False, hdu: HDUSelector = 0,
) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor | None] | None:
    """Decode to VRAM, optionally deriving frame levels in ADU. None selects CPU fallback."""
    if pixel_major and not 2 <= len(paths) <= 32:
        raise ValueError("Packed flat experiments require 2 to 32 frames")
    headers = []
    for index, path in enumerate(paths):
        with (torch.profiler.record_function("stax.fits.read"),
              path.open("rb", buffering=0) as file,
              fits.open(file, memmap=False, do_not_scale_image_data=True) as hdus):
            header, shape, offset = _image_layout(hdus, hdu, file, path)
            if index == 0:
                if not _is_unsigned16(header):
                    if pixel_major:
                        raise ValueError("Packed decoding requires unsigned 16-bit FITS")
                    return None
                if shape[0] * shape[1] >= 2**31:
                    raise ValueError("Each frame must contain fewer than 2**31 pixels")
                words = torch.empty((len(paths), *shape), dtype=torch.uint16, device=device)
                slot = torch.empty(shape, dtype=torch.uint16, device="cpu", pin_memory=True)
                view = memoryview(slot.numpy()).cast("B")
            elif shape != words.shape[1:] or not _is_unsigned16(header):
                raise ValueError("Batch FITS images must have the same shape and dtype")
            headers.append(header)
            if offset is None:
                # Astropy decompresses this frame in storage units. Stage it
                # in FITS byte order for the same unsigned decoder as its peers.
                np.copyto(slot.numpy().view(">i2"), hdus[hdu].data, casting="equiv")
            else:
                _read_payload(file.fileno(), offset, view)
        with torch.profiler.record_function("stax.fits.upload"):
            # A completed copy releases the only host slot for the next read.
            words[index].copy_(slot, non_blocking=False)

    with torch.cuda.device(words.device):
        levels = None
        histogram = (torch.zeros((len(paths), 65536), dtype=torch.int32, device=words.device)
                     if return_levels else None)
        pixels = slot.numel()
        if pixel_major:
            n = len(paths)
            packed = torch.empty((*shape, n), dtype=words.dtype, device=words.device)
            _decode_packed_fits[(triton.cdiv(pixels, 32),)](
                words, packed, histogram, pixels, n, triton.next_power_of_2(n), 32,
            )
            words = packed.permute(2, 0, 1)
        elif return_levels:
            block = 2048
            _decode_fits_u16[(len(paths) * triton.cdiv(pixels, block),)](
                words, pixels, BLOCK=block, histogram=histogram, FRAMES=len(paths),
            )
        else:
            _decode_fits_u16[(triton.cdiv(words.numel(), 512),)](words, words.numel(), BLOCK=512)
        if return_levels:
            levels = _levels_from_histogram(histogram)
    return words, _fits_metadata(paths, headers), levels


def load_fits_host(paths: FitsPath | Iterable[FitsPath], *, hdu: HDUSelector = 0, workers: int = 4,
                   float_units: str = "normalized", pin_memory: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    """Read FITS into a NumPy array, optionally backed by pinned CPU storage."""
    paths, single = _fits_paths(paths, workers)
    first, header = read_fits(paths[0], hdu=hdu, float_units=float_units)
    headers = [header] + [None] * (len(paths) - 1)
    if single:
        values = torch.from_numpy(first).pin_memory().numpy() if pin_memory else first
    else:
        shape = (len(paths), *first.shape)
        if pin_memory:
            values = torch.empty(shape, dtype=torch.from_numpy(first).dtype,
                                 device="cpu", pin_memory=True).numpy()
        else:
            values = np.empty(shape, dtype=first.dtype)
        values[0] = first

        def read_one(item):
            index, path = item
            array, hdr = read_fits(path, hdu=hdu, float_units=float_units)
            if array.shape != first.shape or array.dtype != first.dtype:
                raise ValueError("Batch FITS images must have the same shape and dtype")
            values[index] = array
            headers[index] = hdr

        # One host batch plus bounded reader temporaries, not N arrays plus
        # another complete np.stack copy of the sequence.
        with ThreadPoolExecutor(max_workers=min(workers, len(paths))) as pool:
            list(pool.map(read_one, enumerate(paths[1:], start=1)))
    return values, _fits_metadata(paths, headers)


def write_fits(path: FitsPath, values: np.ndarray, *, header: Header | Mapping[str, Any] | None = None) -> None:
    """Write one host image; preserve usable metadata, regenerate structural cards."""
    values = np.asarray(values)
    if values.ndim != 2 or values.dtype not in (np.uint16, np.float32):
        raise ValueError("Expected a uint16 or normalized float32 [H,W] array")
    cards = Header(header) if header is not None else Header()
    cards.strip()
    for key in ("BSCALE", "BZERO", "BLANK", "DATAMIN", "DATAMAX", "DATASUM", "CHECKSUM",
                "EXTNAME", "EXTVER", "EXTLEVEL"):
        cards.remove(key, remove_all=True, ignore_missing=True)
    fits.PrimaryHDU(values, header=cards).writeto(path, overwrite=True)

"""A small eager API over owned CUDA tensors; host transfers are explicit."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from os import PathLike
from types import MappingProxyType
from typing import Any, ClassVar, Literal, Self, overload

import numpy as np
import torch
from astropy.io.fits import Header

from . import _siril_compat
from ._background import INV_USHORT, estimate_backgrounds_u16
from .io import (
    FitsPath,
    HDUSelector,
    _fits_paths,
    _load_u16_cuda,
    load_fits_host,
    write_fits,
)

__all__ = ["Image", "Sequence"]

type Metadata = Mapping[str, Any]
type Device = str | torch.device


class _CudaPixels[Source: FitsPath | Iterable[FitsPath]]:
    _ndim: ClassVar[int]

    def __init__(self, tensor: torch.Tensor, *, metadata: Metadata | None = None) -> None:
        if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
            raise ValueError("wrap requires a CUDA tensor. Use upload for host data")
        if tensor.dtype not in (torch.uint16, torch.float32):
            raise TypeError("Images must be uint16 ADU or normalized float32")
        if tensor.ndim != self._ndim or not tensor.is_contiguous() or tensor.numel() == 0:
            raise ValueError(f"{type(self).__name__} requires a nonempty contiguous {self._ndim}D tensor")
        if tensor.shape[-2] * tensor.shape[-1] >= 2**31:
            raise ValueError("Each frame must contain fewer than 2**31 pixels")
        self._tensor = tensor
        self._metadata = MappingProxyType(dict(metadata or {}))

    @classmethod
    def wrap(cls, tensor: torch.Tensor, *, metadata: Metadata | None = None) -> Self:
        """Wrap existing CUDA storage without allocating or copying pixels."""
        return cls(tensor, metadata=metadata)

    @classmethod
    def upload(cls, values: np.ndarray | torch.Tensor, *, device: Device = "cuda",
               float_units: str = "normalized", metadata: Metadata | None = None) -> Self:
        """Explicitly upload uint16/float32 host data, or reuse same-device tensors.

        Use ``float_units='adu'`` for float32 ADU input; conversion allocates a
        normalized float32 device result. CUDA tensors in normalized units on
        the requested device keep their original storage.
        """
        if torch.device(device).type != "cuda":
            raise ValueError(f"{cls.__name__} requires a CUDA device")
        if float_units not in ("adu", "normalized"):
            raise ValueError("float_units must be 'adu' or 'normalized'")
        if isinstance(values, np.ndarray):
            values = np.ascontiguousarray(values)
        tensor = torch.as_tensor(values)
        if tensor.dtype not in (torch.uint16, torch.float32):
            raise TypeError("Upload uint16 or float32; cast other types explicitly")
        tensor = tensor.to(device=device)
        if tensor.dtype == torch.float32 and float_units == "adu":
            tensor = tensor * INV_USHORT
        return cls(tensor, metadata=metadata)

    @classmethod
    @overload
    def from_fits(cls, paths: Source, *, hdu: HDUSelector = 0, device: Device = "cuda", workers: int = 4,
                  float_units: str = "normalized", pin_memory: bool = True,
                  return_levels: Literal[False] = False) -> Self: ...

    @classmethod
    @overload
    def from_fits(cls, paths: Source, *, hdu: HDUSelector = 0, device: Device = "cuda", workers: int = 4,
                  float_units: str = "normalized", pin_memory: bool = True,
                  return_levels: Literal[True]) -> tuple[Self, torch.Tensor]: ...

    @classmethod
    def from_fits(cls, paths: Source, *, hdu: HDUSelector = 0, device: Device = "cuda", workers: int = 4,
                  float_units: str = "normalized", pin_memory: bool = True,
                  return_levels: bool = False) -> Self | tuple[Self, torch.Tensor]:
        """Upload one FITS path for Image or an ordered iterable of paths for Sequence.

        Select a 2D image HDU by index, name or (name, version).
        For uint16, return_levels also returns the float64 CUDA levels in ADU.
        """
        single = isinstance(paths, (str, PathLike))
        if single != (cls._ndim == 2):
            raise TypeError("Image.from_fits takes one path. Sequence.from_fits takes an iterable of paths.")
        if torch.device(device).type != "cuda":
            raise ValueError(f"{cls.__name__} requires a CUDA device")
        if float_units not in ("adu", "normalized"):
            raise ValueError("float_units must be 'adu' or 'normalized'")
        inputs: FitsPath | Iterable[FitsPath] = paths
        if pin_memory:
            files, single = _fits_paths(inputs, workers)
            batch = _load_u16_cuda(files, device, hdu=hdu, return_levels=return_levels)
            if batch is not None:
                words, metadata, levels = batch
                image = cls.wrap(words[0] if single else words, metadata=metadata)
                if return_levels:
                    assert levels is not None
                    return image, levels
                return image
            inputs = files[0] if single else files
        with torch.profiler.record_function("stax.fits.read"):
            values, metadata = load_fits_host(inputs, hdu=hdu, workers=workers, float_units=float_units,
                                              pin_memory=pin_memory)
        with torch.profiler.record_function("stax.fits.upload"):
            image = cls.upload(values, device=device, metadata=metadata)
        if return_levels:
            if image.tensor.dtype != torch.uint16:
                raise ValueError("return_levels requires uint16 FITS")
            frames = image.tensor.reshape(-1, image.shape[-2] * image.shape[-1])
            return image, estimate_backgrounds_u16(frames)
        return image

    @property
    def tensor(self) -> torch.Tensor:
        """The underlying CUDA tensor; direct PyTorch/Triton interoperability."""
        return self._tensor

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def __repr__(self) -> str:
        return f"{type(self).__name__}(shape={tuple(self.shape)}, dtype={self.tensor.dtype}, device={self.device})"

    def __dlpack__(self, stream: int | None = None) -> Any:
        return self.tensor.__dlpack__(stream=stream)

    def __dlpack_device__(self) -> tuple[int, int]:
        return self.tensor.__dlpack_device__()

    def to_host(self, *, pin_memory: bool = True) -> np.ndarray:
        """Download pixels into a ready-to-read NumPy array, pinned by default."""
        host = torch.empty_like(self.tensor, device="cpu", pin_memory=pin_memory)
        host.copy_(self.tensor, non_blocking=False)
        return host.numpy()


class Image(_CudaPixels[FitsPath]):
    """A single resident CUDA image with shape [height, width]."""

    _ndim = 2

    def save(self, path: FitsPath) -> None:
        """Explicitly download and write a single image as uncompressed FITS."""
        headers = self.metadata.get("fits_headers", ({},))
        write_fits(path, self.to_host(), header=headers[0] if headers else {})


class Sequence(_CudaPixels[Iterable[FitsPath]]):
    """An ordered, frame-major CUDA batch with shape [frames, height, width]."""

    _ndim = 3

    @classmethod
    def from_images(cls, images: Iterable[Image]) -> Self:
        """Assemble resident images into one contiguous allocation on their device."""
        frames = list(images)
        if not frames or not all(isinstance(image, Image) for image in frames):
            raise ValueError("Supply at least one resident Image")
        first = frames[0]
        if any(frame.shape != first.shape or frame.device != first.device
               or frame.tensor.dtype != first.tensor.dtype for frame in frames):
            raise ValueError("Images must have the same shape, dtype and device")
        metadata: dict[str, Any] = {"frame_metadata": tuple(frame.metadata for frame in frames)}
        for key in ("paths", "fits_headers"):
            if all(key in frame.metadata for frame in frames):
                metadata[key] = tuple(frame.metadata[key][0] for frame in frames)
        if "row_order" in first.metadata and all(
            frame.metadata.get("row_order") == first.metadata["row_order"] for frame in frames
        ):
            metadata["row_order"] = first.metadata["row_order"]
        pixels = torch.stack([frame.tensor for frame in frames])
        return cls.wrap(pixels, metadata=metadata)

    def __len__(self) -> int:
        return self.shape[0]

    @overload
    def __getitem__(self, index: int) -> Image: ...

    @overload
    def __getitem__(self, index: slice) -> Self: ...

    def __getitem__(self, index: int | slice) -> Image | Self:
        """Select a frame or a contiguous range as a storage-sharing view."""
        if not isinstance(index, (int, slice)):
            raise TypeError("Index a sequence by frame number or slice")
        metadata = dict(self.metadata)
        frame_metadata = metadata.pop("frame_metadata", None)
        if frame_metadata is not None:
            if isinstance(index, int):
                metadata = dict(frame_metadata[index]) | metadata
            else:
                metadata["frame_metadata"] = frame_metadata[index]
        for key in ("paths", "fits_headers"):
            if key in self.metadata:
                selected = self.metadata[key][index]
                metadata[key] = (selected,) if isinstance(index, int) else selected
        pixels = self.tensor[index]
        if isinstance(index, int):
            return Image.wrap(pixels, metadata=metadata)
        return type(self).wrap(pixels, metadata=metadata)

    def stack_flats(self, *, levels_adu: torch.Tensor | None = None, reference: int = 0,
                    sigma: float = 3.0, one_warp_per_pixel: bool | None = None) -> Image:
        """Stack raw uint16 flats as in stack_siril_flats.ssf, returning a CUDA Image.

        Supply float64 CUDA levels in ADU to reuse Siril's cached locations.
        By default the frame locations are measured from the resident pixels.
        Batches of 2–256 frames default to one warp per pixel. Larger batches
        use tiled clipping. Set one_warp_per_pixel=False to force tiled clipping.
        """
        if self.tensor.dtype != torch.uint16:
            raise ValueError("Expected a uint16 [N,H,W] flat sequence")
        pixels = _siril_compat.stack_siril_u16(
            self.tensor.flatten(1), levels_adu, sigma=sigma, reference=reference,
            normalization="mul", rejection="winsorized", one_warp_per_pixel=one_warp_per_pixel,
        )
        header = Header()
        header["IMAGETYP"] = "Master Flat"
        header["NCOMBINE"] = len(self)
        header.add_history(f"STAX multiplicative normalization, reference frame {reference} (zero-based)")
        header.add_history(f"Winsorized sigma clipping, low={float(sigma):g}, high={float(sigma):g}")
        inputs = self.metadata.get("fits_headers", ())
        for key in ("BAYERPAT", "XBAYROFF", "YBAYROFF"):
            if inputs and key in inputs[0] and all(h.get(key) == inputs[0][key] for h in inputs):
                header[key] = inputs[0][key]
        metadata = {"fits_headers": (header,), "stack_count": self.shape[0], "normalization": "mul",
                    "rejection": "winsorized", "reference": reference, "sigma": sigma,
                    "row_order": self.metadata.get("row_order", "array")}
        return Image.wrap(pixels.reshape(self.shape[1:]), metadata=metadata)

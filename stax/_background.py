"""GPU reproduction of Siril's scalar normalization estimator for uint16 FITS.

This is a per-frame level, not a spatial gradient/background model. A histogram
of sensor codes lets all subsequent median/MAD passes operate on 65536 weighted
values rather than rereading every pixel. Histogram quantization/interpolation
follows Siril's rt/rt_algo.cc, including float32 percentile thresholds.
"""

import torch
import triton
import triton.language as tl

INV_USHORT = 0.000015259022


@triton.jit
def _histogram_u16(source, hist, pixels, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    frame = tl.program_id(0)
    p = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(source + frame.to(tl.int64) * STRIDE + p, p < pixels, other=0).to(tl.int32)
    tl.atomic_add(hist + frame.to(tl.int64) * 65536 + value, 1, (p < pixels) & (value != 0), sem="relaxed")


def _weighted_median(values, weights):
    n = weights.sum(dim=1, dtype=torch.int64)
    present = weights > 0
    low = torch.where(present, values, float("inf")).amin(dim=1)
    high = torch.where(present, values, -float("inf")).amax(dim=1)
    low = torch.where(n > 0, low, 0.0)
    high = torch.where(n > 0, high, 0.0)
    span = high - low
    bins = torch.minimum(n, torch.full_like(n, 65536))
    scale = torch.where(span > 0, (bins.to(torch.float32) - 1) / span, 1.0)
    index = ((values - low[:, None]) * scale[:, None]).to(torch.int64).clamp_(0, 65535)
    hist = torch.zeros_like(weights)
    hist.scatter_add_(1, index, weights)
    cumulative = hist.cumsum(dim=1, dtype=torch.int64)
    threshold = n.to(torch.float32) * 0.5
    index = torch.searchsorted(cumulative.to(torch.float32), threshold[:, None]).clamp_max_(65535)
    count = cumulative.gather(1, index).squeeze(1)
    bin_count = hist.gather(1, index).squeeze(1)
    c0 = count.to(torch.float32) - threshold
    c1 = threshold - (count - bin_count).to(torch.float32)
    k = index.squeeze(1).to(torch.float32) + 1
    interpolated = (c1 * k + c0 * (k - 1)) / (c0 + c1).clamp_min(1)
    result = (interpolated / scale + low).clamp(min=low, max=high)
    result = torch.where(span == 0, low, result)
    return torch.where(n > 0, result, float("nan"))


@torch.no_grad()
def estimate_backgrounds_u16(frames: torch.Tensor, *, return_diagnostics=False):
    """Estimate Siril additive-normalization levels, returned as float64 ADU.

    Input is contiguous uint16 CUDA ``[frames, pixels]``. Original zero codes
    are excluded, as in Siril. All-zero frames produce NaN. No CPU estimator
    or Siril-generated coefficients are used in this GPU implementation.
    """
    if frames.ndim != 2 or not frames.is_cuda or frames.dtype != torch.uint16 or frames.stride(1) != 1:
        raise ValueError("Expected CUDA uint16 [frames, pixels] with contiguous pixels")
    count, pixels = frames.shape
    if count < 1 or pixels < 1 or pixels >= 2**31:
        raise ValueError("Expected nonempty frames with fewer than 2**31 pixels each")
    histogram = torch.zeros((count, 65536), device=frames.device, dtype=torch.int32)
    _histogram_u16[(count, triton.cdiv(pixels, 4096))](
        frames, histogram, pixels, frames.stride(0), 4096,
    )
    return _levels_from_histogram(histogram, return_diagnostics=return_diagnostics)


def _levels_from_histogram(histogram, *, return_diagnostics=False):
    """Compute Siril locations from CUDA int32 [frames, 65536] nonzero-ADU counts."""
    values = torch.arange(65536, device=histogram.device, dtype=torch.float32)[None, :] * INV_USHORT
    values = values.expand(histogram.shape[0], -1)
    median = _weighted_median(values, histogram)
    mad = _weighted_median(torch.abs(values - median[:, None]), histogram)
    # Siril forms these bounds in double precision, then stores them as float.
    lower = (median.double() - 6 * mad.double()).float()
    upper = (median.double() + 6 * mad.double()).float()
    trimmed = torch.where((values >= lower[:, None]) & (values <= upper[:, None]), histogram, 0)
    location = _weighted_median(values, trimmed)
    levels = location.double() * 65535.0
    if return_diagnostics:
        return levels, {"median_adu": median.double() * 65535.0,
                        "mad_adu": mad.double() * 65535.0,
                        "valid_pixels": histogram.sum(dim=1, dtype=torch.int64),
                        "trimmed_pixels": trimmed.sum(dim=1, dtype=torch.int64)}
    return levels

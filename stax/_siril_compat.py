"""Siril's uint16 mean stacking with sigma or winsorized sigma rejection.

The common path sorts once. Order-dependent, guard-limited pixels are compacted
and replayed on the GPU using Siril's quickselect and stable-compaction order.
No CPU pixel processing is used.
"""

import math
import operator
from functools import cache

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from ._background import INV_USHORT, estimate_backgrounds_u16

INV_USHORT_TL = tl.constexpr(INV_USHORT)
_MAX_WARP_FRAMES = 256  # 256 * 65535 still fits exactly in the FP32 significand.


@cache
def _variance_reciprocals(device):
    """Immutable device table indexed by degrees of freedom for the one-warp kernel."""
    return torch.tensor([0.0] + [1.0 / d for d in range(1, _MAX_WARP_FRAMES)], dtype=torch.float64, device=device)


@triton.jit
def _small_stack_variance(sum_squares, dof, reciprocals):
    reciprocal = tl.load(reciprocals + dof.to(tl.int32))
    quotient = sum_squares * reciprocal
    # Positive nonzero variances of uint16 samples are normal in FP32. Conversion
    # drops 29 FP64 fraction bits; a low field of 2**28 is exactly halfway.
    # The rounded reciprocal and multiply can shift the quotient by a few FP64
    # ulps. Near a midpoint use Siril's division before rounding to FP32.
    fraction = (quotient.to(tl.uint64, bitcast=True) & ((1 << 29) - 1)).to(tl.int32)
    if (tl.abs(fraction - (1 << 28)) <= 4).item():
        quotient = libdevice.div_rn(sum_squares, dof)
    return quotient.to(tl.float32)


@triton.jit
def _mean_adu(total, count, SMALL: tl.constexpr):
    if SMALL:
        # Up to 256 * 65535 ADU and its divisor are exactly representable in FP32.
        return libdevice.div_rn(total.to(tl.float32), tl.maximum(count, 1).to(tl.float32))
    else:
        return libdevice.div_rn(total.to(tl.float64), tl.maximum(count, 1).to(tl.float64)).to(tl.float32)


@triton.jit
def _sd(x, keep, count, AXIS: tl.constexpr, reciprocals=None):
    total = tl.sum(tl.where(keep, x, 0).to(tl.int32), axis=AXIS)
    mean = _mean_adu(total, count, reciprocals is not None)
    d = tl.where(keep, x.to(tl.float32) - tl.expand_dims(mean, AXIS), 0.0)
    squares = d * d
    acc = tl.sum(squares.to(tl.float64), axis=AXIS)
    dof = tl.maximum(count - 1, 1).to(tl.float64)
    if reciprocals is not None:
        variance = _small_stack_variance(acc, dof, reciprocals)
    else:
        variance = libdevice.div_rn(acc, dof).to(tl.float32)
    return tl.sqrt_rn(variance)


@triton.jit
def _normalize(raw, coefficient, MULTIPLY: tl.constexpr):
    if MULTIPLY:
        value = raw.to(tl.float64) * coefficient + 0.5
        return tl.minimum(tl.maximum(value, 0.0), 65535.0).to(tl.int32)
    else:
        return tl.where(raw != 0, tl.minimum(tl.maximum(raw + coefficient, 0), 65535), 0)


@triton.jit
def _winsor_sd(x, keep, count, median, sd, active, AXIS: tl.constexpr, TRACK: tl.constexpr = False,
               reciprocals=None):
    # The scratch values stay clipped between iterations. Only the dispersion
    # estimate uses these values; rejection and averaging use the originals.
    work = x
    refining = active
    more = True
    if TRACK:
        refinements = tl.full(active.shape, 0, tl.int32)
    while more:
        if TRACK:
            refinements += refining.to(tl.int32)
        radius = 1.5 * sd
        low = tl.maximum(0.0, median - radius + 0.5).to(tl.int32)
        high = tl.minimum(65535.0, median + radius + 0.5).to(tl.int32)
        work = tl.minimum(tl.maximum(work, tl.expand_dims(low, AXIS)),
                          tl.expand_dims(high, AXIS))
        updated = 1.134 * _sd(work, keep, count, AXIS, reciprocals)
        # Wider bounds cannot alter an already clamped sample. Only a narrower
        # interval can require another pass with the same median and survivors.
        changed = (sd - updated) > sd * 0.0005
        sd = tl.where(refining, updated, sd)
        refining = refining & changed
        if AXIS == 0:
            more = refining
        else:
            more = tl.sum(refining.to(tl.int32), 0) > 0
    if TRACK:
        return sd, refinements
    else:
        return sd


@triton.jit
def _fast(source, coefficients, output, counts, steps_out, replay_out, pixels,
          N: tl.constexpr, SIGMA: tl.constexpr, CAP: tl.constexpr,
          BP: tl.constexpr, BF: tl.constexpr, MULTIPLY: tl.constexpr, WINSOR: tl.constexpr,
          PACKED: tl.constexpr = False, refinements_out=None,
          PACK_REJECTION_COUNTS: tl.constexpr = False, survivor_bits_out=None, reciprocals=None):
    if reciprocals is not None:
        tl.static_assert(BP == 1 and BF <= 256)
    p = tl.program_id(0) * BP + tl.arange(0, BP)
    f = tl.arange(0, BF)
    if PACKED:
        offset = p[:, None].to(tl.int64) * N + f[None, :]
    else:
        offset = f[None, :].to(tl.int64) * pixels + p[:, None]
    raw = tl.load(source + offset,
                  (p[:, None] < pixels) & (f[None, :] < N), other=0).to(tl.int32)
    coefficient = tl.load(coefficients + f, f < N, other=0)
    x = _normalize(raw, coefficient[None, :], MULTIPLY)
    valid = (x != 0) & (f[None, :] < N) & (p[:, None] < pixels)
    count = tl.sum(valid.to(tl.int32), axis=1)
    x = tl.sort(tl.where(valid, x, 65536), dim=1)
    keep = f[None, :] < count[:, None]
    first = tl.full((BP,), 0, tl.int32)
    rejected = tl.full((BP,), 0, tl.int32)
    steps = tl.full((BP,), 0, tl.int32)
    replay = tl.full((BP,), False, tl.int1)
    active = (count > 1) & (p < pixels)
    iteration = 0
    while (iteration < CAP) & (tl.sum(active.to(tl.int32), 0) > 0):
        sd = _sd(x, keep, count, 1, reciprocals)
        lo = tl.minimum(first + tl.maximum(count - 1, 0) // 2, BF - 1)
        hi = tl.minimum(first + count // 2, BF - 1)
        a = tl.gather(x, lo[:, None], 1).reshape((BP,)).to(tl.float32)
        b = tl.gather(x, hi[:, None], 1).reshape((BP,)).to(tl.float32)
        median = (a + b) * 0.5
        if WINSOR:
            if refinements_out is not None:
                sd, refinements = _winsor_sd(x, keep, count, median, sd, active, 1, True, reciprocals)
                tl.store(refinements_out + iteration * pixels + p, refinements, p < pixels)
            else:
                sd = _winsor_sd(x, keep, count, median, sd, active, 1, reciprocals=reciprocals)
        bound = sd * SIGMA
        if PACK_REJECTION_COUNTS:
            tl.static_assert(N <= 32)
            delta = x.to(tl.float32) - median[:, None]
            candidates = keep & active[:, None] & (tl.abs(delta) > bound[:, None])
            low_reject = candidates & (delta < 0)
            # At most 32 votes: neither the total nor the low count can overflow a byte.
            votes = tl.sum(candidates.to(tl.int32) + (low_reject.to(tl.int32) << 8), 1)
            proposed = votes & 255
        else:
            low_reject = keep & ((median[:, None] - x.to(tl.float32)) > bound[:, None])
            high_reject = keep & ((x.to(tl.float32) - median[:, None]) > bound[:, None])
            candidates = (low_reject | high_reject) & active[:, None]
            proposed = tl.sum(candidates.to(tl.int32), 1)
        allowed = tl.maximum(count - rejected - 4, 0)
        ambiguous = active & (allowed > 0) & (proposed > allowed)
        replay |= ambiguous
        accept = active & ~ambiguous & (allowed > 0)
        remove = candidates & accept[:, None]
        if PACK_REJECTION_COUNTS:
            first += tl.where(accept, votes >> 8, 0)
            nremove = tl.where(accept, proposed, 0)
        else:
            first += tl.sum((remove & low_reject).to(tl.int32), 1)
            nremove = tl.sum(remove.to(tl.int32), 1)
        keep &= ~remove
        count -= nremove
        rejected += nremove
        steps += active.to(tl.int32)
        active = active & ~ambiguous & (nremove > 0) & (count > 3)
        iteration += 1
    total = tl.sum(tl.where(keep, x, 0), 1)
    mean = _mean_adu(total, count, reciprocals is not None)
    result = tl.minimum(tl.maximum(mean * INV_USHORT_TL, 0.0), 1.0)
    tl.store(output + p, result, p < pixels)
    tl.store(counts + p, count, p < pixels)
    tl.store(steps_out + p, steps, p < pixels)
    tl.store(replay_out + p, replay, p < pixels)
    if survivor_bits_out is not None:
        tl.static_assert(BF <= 32)
        bits = tl.reduce_or(keep.to(tl.uint32) << f[None, :], 1)
        tl.store(survivor_bits_out + p, bits, p < pixels)


@triton.jit
def _at(x, index, f):
    return tl.sum(tl.where(f == index, x, 0), axis=0)


@triton.jit
def _quickmedian(x, n, f, BF: tl.constexpr):
    if n < 9:
        x = tl.sort(tl.where(f < n, x, 65536), dim=0)
    else:
        left = 0
        right = n - 1
        k = n // 2
        while left < right:
            pivot_index = (left + right) // 2
            pivot = _at(x, pivot_index, f)
            right_value = _at(x, right, f)
            x = tl.where(f == pivot_index, right_value, x)
            x = tl.where(f == right, pivot, x)
            pi = left
            for j in range(left, right):
                value = _at(x, j, f)
                if value < pivot:
                    old = _at(x, pi, f)
                    x = tl.where(f == pi, value, x)
                    x = tl.where(f == j, old, x)
                    pi += 1
            at_pi = _at(x, pi, f)
            x = tl.where(f == right, at_pi, x)
            x = tl.where(f == pi, pivot, x)
            if pi < k:
                left = pi + 1
            else:
                right = pi
    lo = tl.maximum(n - 1, 0) // 2
    hi = n // 2
    median = (_at(x, lo, f).to(tl.float32) + _at(x, hi, f).to(tl.float32)) * 0.5
    return x, median


@triton.jit
def _replay(source, coefficients, indices, output, counts, steps_out, pixels,
            N: tl.constexpr, SIGMA: tl.constexpr, CAP: tl.constexpr, BF: tl.constexpr,
            MULTIPLY: tl.constexpr, WINSOR: tl.constexpr, PACKED: tl.constexpr = False):
    p = tl.load(indices + tl.program_id(0))
    f = tl.arange(0, BF)
    if PACKED:
        offset = p * N + f
    else:
        offset = f.to(tl.int64) * pixels + p
    raw = tl.load(source + offset, f < N, other=0).to(tl.int32)
    coefficient = tl.load(coefficients + f, f < N, other=0)
    x = _normalize(raw, coefficient, MULTIPLY)
    valid = (f < N) & (x != 0)
    n = tl.sum(valid.to(tl.int32), 0)
    keys = tl.where(valid, f, f + BF) * 65536 + x
    x = tl.sort(keys, dim=0) & 65535
    x, median = _quickmedian(x, n, f, BF)
    rejected = 0
    iteration = 0
    running = n > 1
    while running & (iteration < CAP):
        sd = _sd(x, f < n, n, 0)
        if iteration > 0:
            x, median = _quickmedian(x, n, f, BF)
        if WINSOR:
            sd = _winsor_sd(x, f < n, n, median, sd, running, 0)
        bound = sd * SIGMA
        candidate = (f < n) & (((median - x.to(tl.float32)) > bound) |
                               ((x.to(tl.float32) - median) > bound))
        ordinal = tl.cumsum(candidate.to(tl.int32), 0)
        remove = candidate & (ordinal <= tl.maximum(n - rejected - 4, 0))
        keep = (f < n) & ~remove
        kept = tl.sum(keep.to(tl.int32), 0)
        rejected += n - kept
        keys = tl.where(keep, f, f + BF) * 65536 + tl.minimum(x, 65535)
        x = tl.sort(keys, dim=0) & 65535
        running = (kept != n) & (kept > 3)
        n = kept
        iteration += 1
    total = tl.sum(tl.where(f < n, x, 0), 0)
    mean = libdevice.div_rn(total.to(tl.float64), tl.maximum(n, 1).to(tl.float64)).to(tl.float32)
    tl.store(output + p, tl.minimum(tl.maximum(mean * INV_USHORT_TL, 0.0), 1.0))
    tl.store(counts + p, n)
    tl.store(steps_out + p, iteration)


@torch.no_grad()
def stack_siril_u16(frames, levels=None, *, sigma=3.0, reference=0, normalization="add",
                     rejection="sigma", max_iterations=None, return_profile=False,
                     one_warp_per_pixel=None):
    """Stack CUDA uint16 [N, pixels] with Siril's uint16 semantics.

    Frame-major and packed pixel-major strides are supported. By default, batches
    of up to 256 frames use one warp per pixel and larger batches use tiled clipping.
    Set one_warp_per_pixel=False to force tiled clipping.

    With ``normalization="add"`` (default), estimate missing levels and round
    pixels after additive normalization relative to the selected reference.
    For raw flats use ``normalization="mul", rejection="winsorized"``. Frame
    levels are in ADU and may be supplied from a Siril sequence cache.
    Use ``normalization="none"`` for master darks: preserve dark signal and bias,
    with no background estimation or subtraction. Rejection
    uses float32 square roots/thresholds, double accumulation, Siril's cumulative
    rejection safeguard, zero exclusion, and float32 normalized/clamped output.
    """
    if frames.ndim != 2 or not frames.is_cuda or frames.dtype != torch.uint16:
        raise ValueError("Expected CUDA uint16 [frames, pixels]")
    n, pixels = frames.shape
    packed = not frames.is_contiguous()
    if packed and frames.stride() != (1, n):
        raise ValueError("Expected frame-major or packed pixel-major storage")
    if one_warp_per_pixel is None:
        one_warp_per_pixel = n <= _MAX_WARP_FRAMES
    if one_warp_per_pixel and n > _MAX_WARP_FRAMES:
        raise ValueError(f"One warp per pixel supports at most {_MAX_WARP_FRAMES} frames")
    if not 2 <= n <= 1024 or not 0 <= reference < n or pixels < 1:
        raise ValueError("Invalid sequence dimensions or reference")
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("Invalid sigma")
    if rejection not in ("sigma", "winsorized"):
        raise ValueError("rejection must be 'sigma' or 'winsorized'")
    if normalization == "none":
        if levels is not None:
            raise ValueError("levels cannot be supplied with normalization='none'")
        coefficients = torch.zeros(n, device=frames.device, dtype=torch.int32)
    elif normalization in ("add", "mul"):
        if levels is None:
            levels = estimate_backgrounds_u16(frames.contiguous())
        if levels.shape != (n,) or levels.device != frames.device or levels.dtype != torch.float64:
            raise ValueError("Expected float64 levels [frames] on the same CUDA device")
        if not torch.isfinite(levels).all().item():
            raise ValueError("Background estimation failed: a frame has no valid normalization level")
        if normalization == "mul":
            coefficients = torch.where(levels != 0, levels[reference] / levels, 1.0)
        else:
            coefficients = torch.floor(levels[reference] - levels + 0.5).to(torch.int32)
    else:
        raise ValueError("normalization must be 'add', 'mul' or 'none'")
    output = torch.empty(pixels, device=frames.device, dtype=torch.float32)
    counts = torch.empty(pixels, device=frames.device, dtype=torch.int32)
    steps = torch.empty(pixels, device=frames.device, dtype=torch.int16)
    replay = torch.empty(pixels, device=frames.device, dtype=torch.bool)
    if max_iterations is None:
        cap = n
    else:
        try:
            cap = min(n, operator.index(max_iterations))
        except TypeError as error:
            raise ValueError("max_iterations must be a positive integer or None") from error
        if cap < 1:
            raise ValueError("max_iterations must be positive")
    bf = triton.next_power_of_2(n)
    bp = 1 if one_warp_per_pixel else min(32, max(1, 1024 // bf))
    multiply = normalization == "mul"
    winsor = rejection == "winsorized"
    _fast[(triton.cdiv(pixels, bp),)](
        frames, coefficients, output, counts, steps, replay, pixels, n, sigma, cap,
        bp, bf, multiply, winsor, packed,
        reciprocals=_variance_reciprocals(frames.device) if one_warp_per_pixel else None,
        num_warps=1 if one_warp_per_pixel else 4, enable_fp_fusion=False,
    )
    indices = torch.nonzero(replay).flatten()
    if indices.numel():
        _replay[(indices.numel(),)](
            frames, coefficients, indices, output, counts, steps, pixels, n, sigma,
            cap, bf, multiply, winsor, packed, num_warps=4, enable_fp_fusion=False,
        )
    if return_profile:
        return output, {"survivors": counts, "iterations": steps,
                        "replayed_pixels": indices.numel(), "levels": levels}
    return output

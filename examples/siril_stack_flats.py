# SPDX-License-Identifier: GPL-3.0-or-later
"""Stack the selected, unregistered uint16 flats and save a master in Siril's working directory."""

import argparse
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import sirilpy
import torch

from stax import Sequence


def selected_flats(siril: sirilpy.SirilInterface) -> tuple[list[Path], int, list[float | None]]:
    seq = siril.get_seq()
    if seq is None or seq.type != sirilpy.SequenceType.SEQ_REGULAR or seq.nb_layers != 1:
        raise ValueError("Open a single-channel, regular FITS flat sequence in Siril.")
    if any(reg is not None for layer in seq.regparam for reg in layer):
        raise ValueError("STAX expects unregistered flat frames.")
    selected = [i for i, frame in enumerate(seq.imgparam) if frame.incl]
    reference = selected.index(seq.reference_image) if seq.reference_image in selected else 0
    paths, cached = [], []
    for i in selected:
        name = siril.get_seq_frame_filename(i)
        if not name:
            raise ValueError(f"Siril could not resolve the filename for frame {i + 1}.")
        paths.append(Path(name))
        stats = seq.stats[0][i]
        level = stats.location if stats is not None else math.nan
        cached.append(level if math.isfinite(level) and level >= 0 else None)
    return paths, reference, cached


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", help="Stack all matching files in filename order, with fresh levels")
    parser.add_argument("--output", type=Path, default=Path("stax_master_flat.fits"))
    args = parser.parse_args()

    siril = sirilpy.SirilInterface()
    siril.connect()
    try:
        wd = Path(siril.get_siril_wd())
        output = wd / args.output
        if output.exists():
            raise FileExistsError(f"Choose a new output filename. {output} already exists.")
        if output.suffix.lower() not in {".fit", ".fits", ".fts"}:
            raise ValueError("Choose an uncompressed FITS output name (.fit, .fits or .fts).")
        if args.pattern:
            paths = sorted(wd.glob(args.pattern))
            reference, cached = 0, [None] * len(paths)
        else:
            if not siril.is_sequence_loaded():
                raise ValueError("Load a flat sequence in Siril or pass --pattern 'flat_?????.fits'.")
            paths, reference, cached = selected_flats(siril)
        if not 2 <= len(paths) <= 1024:
            raise ValueError(f"Stacking requires 2–1024 flat frames. Found {len(paths)}.")

        if None in cached:
            flats, levels = Sequence.from_fits(paths, return_levels=True)
            indices = [i for i, level in enumerate(cached) if level is not None]
            if indices:
                levels[indices] = levels.new_tensor([cached[i] for i in indices])
        else:
            flats = Sequence.from_fits(paths)
            levels = torch.tensor(cached, device=flats.device, dtype=torch.float64)
        master = flats.stack_flats(levels_adu=levels, reference=reference)
        # Publish the completed FITS atomically, including when two scripts choose the same name.
        with TemporaryDirectory(dir=output.parent) as tmp:
            staged = Path(tmp) / "master.fits"
            master.save(staged)
            os.link(staged, output)
        siril.log(f"STAX stacked {len(paths)} frames into {output}")
    finally:
        siril.disconnect()


if __name__ == "__main__":
    main()

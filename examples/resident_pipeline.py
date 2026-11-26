"""Build a GPU-resident master flat from one FITS HDU per input file."""

import argparse
from pathlib import Path

from stax import Sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--hdu", default="0", help="Image HDU index or extension name")
    parser.add_argument("--output", type=Path, help="Download and write the master flat")
    args = parser.parse_args()
    selector = int(args.hdu) if args.hdu.isdecimal() else args.hdu
    flats, levels = Sequence.from_fits(args.paths, hdu=selector, return_levels=True)
    master = flats.stack_flats(levels_adu=levels)
    print(f"Stacked {len(flats)} frames into {master}")
    if args.output:
        master.save(args.output)


if __name__ == "__main__":
    main()

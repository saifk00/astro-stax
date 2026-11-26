# astro-stax

[![Tests on main](https://github.com/saifk00/astro-stax/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/saifk00/astro-stax/actions/workflows/tests.yml?query=branch%3Amain)

STAX builds a master flat from astrophotography flat frames on an NVIDIA GPU.
Its Python API implements Siril’s average stack with multiplicative input
normalization, winsorized sigma clipping and float32 output. The result stays
on the GPU until you download it or write a FITS file.

## Install

Use Python 3.13 on Linux x86-64 and an NVIDIA driver compatible with the CUDA
build of PyTorch. This release was tested on an RTX 5070 Ti with PyTorch 2.15
(CUDA 13.2) and Triton 3.9. Install from the repository in your environment:

```sh
python -m pip install git+https://github.com/saifk00/astro-stax.git
python -c "import torch; print(torch.cuda.is_available())"
```

The distribution name is **astro-stax**. The Python import name is **stax**.
The CUDA check should print `True`. Triton compiles the kernels on first use.

## Stack a set of flats

```python
from pathlib import Path
from stax import Sequence

paths = sorted(Path("flats").glob("flat_?????.fits"))
flats, levels = Sequence.from_fits(paths, return_levels=True)
master = flats.stack_flats(levels_adu=levels, reference=0, sigma=3.0)
master.save("master_flat.fits")
```

Use 2–1024 equally sized, single-channel uint16 frames. The input sequence,
working buffers and master must fit in free GPU memory. Astropy handles FITS
headers and HDU selection. For an image extension, pass e.g. `hdu="SCI"`.
For ordinary unsigned-16 FITS the loader reuses a pinned input buffer and
computes frame levels during GPU decoding. Frames already on the device can
be supplied with `Sequence.wrap(cuda_tensor)` or `Sequence.from_images(images)`.

`reference` indexes the ordered inputs from zero. `sigma` sets both rejection
thresholds. To reproduce a Siril stack that reused saved sequence statistics,
pass the same locations in ADU as a float64 CUDA tensor to `levels_adu`.
Siril rounds these statistics when saving the sequence. Freshly measured levels
can therefore produce different rounded pixels. The master contains normalized
float32 values. Access them through `master.tensor` on the GPU or
`master.to_host()` as a NumPy array. `master.save()` overwrites its destination.

## Run from Siril

Install the package with the Python interpreter in Siril’s virtual environment.
On a default Linux installation:

```sh
~/.local/share/siril/venv/bin/python -m pip install git+https://github.com/saifk00/astro-stax.git
```

Copy `examples/siril_stack_flats.py` from the source distribution into a
configured Siril script directory. The wheel also installs the script as
`stax/siril_stack_flats.py` in site-packages. Open an unregistered regular FITS
flat sequence and run this in Siril’s console:

```text
pyscript siril_stack_flats.py
```

The script reads the original FITS paths and uploads the pixels directly from its
own process. It uses the included frames, the selected reference (or the first
included frame), and available cached levels. It writes a new
`stax_master_flat.fits` in Siril’s working directory. Open it with
`load stax_master_flat.fits`. To run headlessly, specify a filename pattern
relative to Siril’s working directory:

```text
pyscript siril_stack_flats.py --pattern flat_?????.fits --output batch_master.fits
```

Pattern mode uses every match in filename order, the first frame as reference,
and freshly computed levels. Keep successive GPU operations inside the same
script to retain device allocations.

The script was exercised through `pyscript` on Siril 1.5.0-dev (9d9577762)
with sirilpy 1.1.15. On 21 real 8280 × 5520 flat frames all 45,705,600
float32 output pixels matched a fresh native Siril stack bit for bit.

## Development

From the checkout, run `uv sync --locked`, then
`uv run python -m unittest discover -s tests`. Build the source distribution
and wheel with `uv build`. The resulting wheel is
`dist/astro_stax-0.1.0-py3-none-any.whl`.

astro-stax is distributed under GPL-3.0-or-later.

## Consumer GPU benchmarks

Run the same workload on a Linux or WSL2 desktop with:

```sh
uv run --locked python scripts/benchmark_geforce.py --output geforce.json
```

The default sweep uses 8, 21, 64 and 128 deterministic synthetic 24 MP flats
and needs about 6 GiB of scratch disk in the current directory. Each case
gets one warm-up and three timed runs. Resident time covers the stack with
inputs and normalization levels already on the GPU. End-to-end time includes
reading FITS, computing levels, stacking, downloading and writing the master
with a warm filesystem cache. The JSON records individual timings, peak tensor
memory, hardware, commit and a checksum of the master pixels. An out-of-memory
case is recorded at its original frame count.

The GeForce workflow runs Sundays at 06:17 UTC and supports manual runs on
`main`. To enable a card, register an isolated Linux x64 runner with a model
label, then add that label to the repository variable `GEFORCE_RUNNERS`
(a JSON array such as `["rtx-3060-12gb", "rtx-4060-8gb"]`). Only list
available runners. Use dedicated environments for self-hosted Actions on a
public repository. Each run uploads one JSON artifact per card. The two native
Siril comparison tests additionally require the reference executables in the
test checkout.

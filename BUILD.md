# Building BCFP

BCFP has a C++ core (`src/bcfp.cpp`, `src/bcfp_pybind.cpp`) with pybind11 bindings that links
against **RDKit's C++ headers and libraries**. The plain `pip install rdkit` wheel is runtime-only
(no C++ headers), so it cannot build this extension — conda-forge's `librdkit-dev` provides them.

The two non-RDKit hash functions are **self-contained**, so RDKit is the *only* external build
dependency:
- **xxhash** — header-only (`XXH_INLINE_ALL`); nothing to install or link.
- **blake3** — the portable C source is **vendored** under `src/vendor/blake3/` (CC0/Apache-2.0)
  and compiled directly into the extension.

## Quick start (recommended)

```bash
# 1. Create the environment — RDKit 2026.03 + C++ headers + build toolchain, one command
conda env create -f environment.yml      # or: mamba env create -f environment.yml
conda activate bcfp

# 2. Build + install (editable)
pip install -e .

# 3. Verify
python -c "import bcfp; from bcfp.fingerprints import FingerprintGenerator; print('bcfp OK')"
pytest -q
```

`setup.py` auto-detects RDKit from the active conda env (`$CONDA_PREFIX` / `sys.prefix`) — no paths
to edit. An `-rpath` to the env's `lib/` is baked in, so RDKit's dylibs load at import time without
any `DYLD_LIBRARY_PATH` / `LD_LIBRARY_PATH` juggling.

## How detection works

`setup.py` resolves the RDKit prefix in order: `RDKIT_PREFIX` (explicit override) → `CONDA_PREFIX`
(active env) → `sys.prefix` (the interpreter's own prefix — correct for a conda env python even
when it isn't `conda activate`-d). It handles both the conda-forge `include/rdkit/GraphMol/...`
layout and the plain `include/GraphMol/...` layout.

## Requirements

- A **C++20** compiler (clang on macOS, gcc/clang on Linux). RDKit 2026.03's headers use C++20
  features (`constexpr virtual`, `constexpr` destructors), so C++17 no longer compiles.
- RDKit **2026.03** (what CI builds against); 2025.3+ is expected to work.
- Tested on macOS (Apple Silicon) and Linux x86-64.

## Custom RDKit location

```bash
export RDKIT_PREFIX=/path/to/rdkit/prefix   # must contain include/rdkit/ and lib/
pip install -e .
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ERROR: RDKit C++ headers not found` | Missing `librdkit-dev`, or not in the conda env / used the pip `rdkit` wheel. Re-create from `environment.yml`, `conda activate bcfp`, or set `RDKIT_PREFIX`. |
| `fatal error: 'boost/...' file not found` | Missing `libboost-devel`. Re-create the env from `environment.yml`. |
| `invalid argument '-std=c++20' not allowed with 'C'` | The custom `build_ext` in `setup.py` strips C++ flags from the vendored blake3 `.c` files — ensure you're using this repo's `setup.py`. |
| `error: ... cannot be declared constexpr` (in `Geometry/point.h`) | Compiling RDKit 2026.03 headers as C++17. `setup.py` already sets `cxx_std=20`; ensure a C++20 toolchain (`cxx-compiler`). |
| `ImportError: library not loaded ... libRDKit*.dylib` | Import from the same conda env you built in; the baked-in `-rpath` handles this. |

## BLAKE3 performance note

The vendored blake3 is built in **portable** mode (no SIMD) for a simple, dependency-free build
that works on any CPU. Fingerprint hashing is not the bottleneck, so this is fine in practice. To
use the SIMD implementations, add the relevant `blake3_*` SIMD sources and drop the `BLAKE3_NO_*`
defines in `setup.py`.

# Building BCFP

BCFP has a C++ core (`src/bcfp.cpp`, `src/bcfp_pybind.cpp`) with nanobind bindings that links
against **RDKit's C++ headers and libraries**. The plain `pip install rdkit` wheel is runtime-only
(no C++ headers), so it cannot build this extension — conda-forge's `librdkit-dev` provides them.

The two non-RDKit hash functions do not require linked libraries:
- **xxhash** — header-only (`XXH_INLINE_ALL`); `environment.yml` supplies `xxhash.h`.
- **blake3** — the portable C source is **vendored** under `src/vendor/blake3/` (CC0/Apache-2.0)
  and compiled directly into the extension.

## Quick start (recommended)

```bash
# 1. Create the environment — RDKit 2026.09.2 + C++ headers + build toolchain
conda env create -f environment.yml      # or: mamba env create -f environment.yml
conda activate bcfp

# 2. Build + install (editable)
pip install -e .

# 3. Verify
python -c "import bcfp; from bcfp.fingerprints import FingerprintGenerator; print('bcfp OK')"
pytest -q
```

The CMake build auto-detects RDKit from the active conda env (`$CONDA_PREFIX` / `sys.prefix`) — no paths
to edit. An `-rpath` to the env's `lib/` is baked in, so RDKit's dylibs load at import time without
any `DYLD_LIBRARY_PATH` / `LD_LIBRARY_PATH` juggling.

## How detection works

CMake resolves the RDKit prefix in order: `RDKIT_PREFIX` (explicit override) → `CONDA_PREFIX`
(active env) → `sys.prefix` (the interpreter's own prefix — correct for a conda env python even
when it isn't `conda activate`-d). It handles both the conda-forge `include/rdkit/GraphMol/...`
layout and the plain `include/GraphMol/...` layout.

## Requirements

- A **C++20** compiler (clang on macOS, gcc/clang on Linux).
- RDKit **2026.09.2**.
- nanobind **2.4–2.x**, CMake, Ninja, and scikit-build-core (all provided by `environment.yml`).
- Tested on macOS (Apple Silicon) and Linux x86-64.

## RDKit's nanobind wrappers

RDKit 2026.09 includes its new nanobind Python wrappers, but they remain a source-build option.
The conda-forge `rdkit=2026.09.2` package currently links to `libboost-python`; it is still useful
as the reproducible default runtime and as the source of matching C++ development libraries.

To exercise BCFP with RDKit's nanobind wrappers, build the `Release_2026_09_2` RDKit tag with:

```bash
cmake -S /path/to/rdkit -B /path/to/rdkit-build \
  -DRDK_BUILD_BOOST_PYTHON_WRAPPERS=OFF \
  -DRDK_BUILD_NANOBIND_WRAPPERS=ON \
  -DPython_EXECUTABLE="$CONDA_PREFIX/bin/python"
cmake --build /path/to/rdkit-build --parallel
cmake --install /path/to/rdkit-build
```

With RDKit's default in-tree install, put `/path/to/rdkit` first on `PYTHONPATH` and its `lib/`
directory on `LD_LIBRARY_PATH` (Linux) or `DYLD_LIBRARY_PATH` (macOS). BCFP can remain linked to the
matching conda-forge 2026.09.2 C++ libraries: the serialized `Mol.ToBinary()` boundary deliberately
does not exchange C++ object pointers between the two extensions.

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
| `fatal error: 'xxhash.h' file not found` | Missing the header-only `xxhash` package. Re-create the env from `environment.yml`. |
| `Could not find nanobind` | Install the build requirements from `environment.yml`, or let pip create an isolated build environment. |
| `error: ... cannot be declared constexpr` (in `Geometry/point.h`) | Compiling RDKit headers as C++17. `CMakeLists.txt` requests C++20; ensure a C++20 toolchain (`cxx-compiler`). |
| `ImportError: library not loaded ... libRDKit*.dylib` | Import from the same conda env you built in; the baked-in `-rpath` handles this. |

## BLAKE3 performance note

The vendored blake3 is built in **portable** mode (no SIMD) for a simple, dependency-free build
that works on any CPU. Fingerprint hashing is not the bottleneck, so this is fine in practice. To
use the SIMD implementations, add the relevant `blake3_*` SIMD sources and drop the `BLAKE3_NO_*`
defines in `CMakeLists.txt`.

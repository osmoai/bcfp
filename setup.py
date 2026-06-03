#!/usr/bin/env python3
"""Build script for BCFP — C++ core (src/bcfp.cpp) + pybind11 bindings.

Links against RDKit's C++ headers/libraries (conda-forge `librdkit-dev` + `libboost-devel`;
see environment.yml and BUILD.md). The two non-RDKit hashes are self-contained:
  * xxhash  — header-only (XXH_INLINE_ALL), no library to install;
  * blake3  — vendored portable C source under src/vendor/blake3 (CC0/Apache-2.0), compiled in.
So the only external build dependency is RDKit. Project metadata lives in pyproject.toml.
"""
import os
import sys

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext as _pybind_build_ext
import pybind11


class build_ext(_pybind_build_ext):
    """pybind11's build_ext, but don't pass C++ flags (-std=c++NN) to vendored C sources
    (the blake3 *.c files), which clang rejects for C."""

    def build_extensions(self):
        compiler = self.compiler
        original_compile = compiler._compile

        def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if src.endswith(".c"):
                extra_postargs = [a for a in extra_postargs
                                  if not (a.startswith("-std=c++") or a.startswith("-std=gnu++"))]
            return original_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

        compiler._compile = _compile
        try:
            super().build_extensions()
        finally:
            compiler._compile = original_compile


def find_rdkit():
    """Return (prefix, include_dirs, lib_dir) for an RDKit install with C++ headers.

    Order: RDKIT_PREFIX env var, then CONDA_PREFIX, then sys.prefix (so it works whether or
    not the conda env is `conda activate`-d). Handles both include/rdkit/GraphMol/... and
    include/GraphMol/... layouts.
    """
    for prefix in (os.environ.get("RDKIT_PREFIX"), os.environ.get("CONDA_PREFIX"), sys.prefix):
        if not prefix:
            continue
        inc = os.path.join(prefix, "include")
        if os.path.exists(os.path.join(inc, "rdkit", "GraphMol", "ROMol.h")):
            return prefix, [inc, os.path.join(inc, "rdkit")], os.path.join(prefix, "lib")
        if os.path.exists(os.path.join(inc, "GraphMol", "ROMol.h")):
            return prefix, [inc], os.path.join(prefix, "lib")
    sys.exit(
        "\nERROR: RDKit C++ headers not found.\n"
        "BCFP's C++ core links against RDKit's headers + libraries, which the plain\n"
        "`pip install rdkit` wheel does NOT ship. Use a conda-forge environment:\n\n"
        "    conda env create -f environment.yml   # or: mamba env create -f environment.yml\n"
        "    conda activate bcfp\n"
        "    pip install -e .\n\n"
        "Or set RDKIT_PREFIX to an RDKit install containing include/rdkit/ and lib/.\n"
    )


prefix, rdkit_includes, lib_dir = find_rdkit()
print(f"[bcfp] building against RDKit in: {prefix}")

HERE = os.path.dirname(os.path.abspath(__file__))
BLAKE3 = os.path.join(HERE, "src", "vendor", "blake3")

# Force the portable BLAKE3 implementation (no SIMD source files / arch flags) so the build is
# simple and works on any CPU. Fingerprint hashing is not the bottleneck, so this is fine.
BLAKE3_PORTABLE = [("BLAKE3_USE_NEON", "0"), ("BLAKE3_NO_SSE2", "1"),
                   ("BLAKE3_NO_SSE41", "1"), ("BLAKE3_NO_AVX2", "1"), ("BLAKE3_NO_AVX512", "1")]

RDKIT_LIBS = ["RDKitGraphMol", "RDKitSmilesParse", "RDKitDataStructs", "RDKitFingerprints"]

ext_modules = [
    Pybind11Extension(
        "_bcfp",
        sources=[
            "src/bcfp.cpp",
            "src/bcfp_pybind.cpp",
            "src/vendor/blake3/blake3.c",
            "src/vendor/blake3/blake3_dispatch.c",
            "src/vendor/blake3/blake3_portable.c",
        ],
        include_dirs=[pybind11.get_include(), *rdkit_includes,
                      BLAKE3, os.path.join(prefix, "include")],  # env include for xxhash.h
        libraries=RDKIT_LIBS,                                    # xxhash header-only, blake3 vendored
        library_dirs=[lib_dir],
        define_macros=BLAKE3_PORTABLE,
        extra_link_args=[f"-Wl,-rpath,{lib_dir}"],
        language="c++",
        cxx_std=20,  # RDKit 2026.03 headers require C++20
        extra_compile_args=["-O3"] if sys.platform != "win32" else ["/O2"],
    ),
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})

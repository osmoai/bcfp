# Vendored BLAKE3 (portable C)

These are the **portable** C source files of the official
[BLAKE3](https://github.com/BLAKE3-team/BLAKE3) reference implementation, vendored so that BCFP is
self-contained and needs no external BLAKE3 library to build (the conda-forge `blake3` package is
the Python binding, not the C library).

- **Upstream:** https://github.com/BLAKE3-team/BLAKE3
- **Version:** v1.5.5 (the `c/` directory)
- **Files:** `blake3.h`, `blake3_impl.h`, `blake3.c`, `blake3_dispatch.c`, `blake3_portable.c`
- **License:** dual CC0-1.0 / Apache-2.0 (see `LICENSE_CC0`). Permissive; redistribution is allowed.

`setup.py` compiles these with `-DBLAKE3_USE_NEON=0 -DBLAKE3_NO_SSE2 -DBLAKE3_NO_SSE41
-DBLAKE3_NO_AVX2 -DBLAKE3_NO_AVX512`, i.e. the **portable** path only (no SIMD sources / arch
flags). Hashing is not the fingerprinting bottleneck, so this keeps the build simple and
CPU-independent. To enable SIMD, add the corresponding `blake3_*` SIMD sources and drop the
`BLAKE3_NO_*` defines.

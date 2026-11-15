cmake_minimum_required(VERSION 3.15...3.31)
project(bcfp LANGUAGES C CXX)

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

if(DEFINED ENV{RDKIT_PREFIX})
  set(RDKIT_PREFIX "$ENV{RDKIT_PREFIX}")
elseif(DEFINED ENV{CONDA_PREFIX})
  set(RDKIT_PREFIX "$ENV{CONDA_PREFIX}")
else()
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import sys; print(sys.prefix)"
    OUTPUT_VARIABLE RDKIT_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
endif()

find_path(
  RDKIT_INCLUDE_DIR
  NAMES GraphMol/ROMol.h
  HINTS "${RDKIT_PREFIX}/include/rdkit" "${RDKIT_PREFIX}/include"
  REQUIRED
)
find_path(
  XXHASH_INCLUDE_DIR
  NAMES xxhash.h
  HINTS "${RDKIT_PREFIX}/include"
  REQUIRED
)

foreach(component GraphMol SmilesParse DataStructs Fingerprints RDGeneral)
  find_library(
    RDKIT_${component}_LIBRARY
    NAMES RDKit${component}
    HINTS "${RDKIT_PREFIX}/lib"
    REQUIRED
  )
  list(APPEND RDKIT_LIBRARIES "${RDKIT_${component}_LIBRARY}")
endforeach()

set(BLAKE3_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/vendor/blake3")
set(BLAKE3_SOURCES
  "${BLAKE3_DIR}/blake3.c"
  "${BLAKE3_DIR}/blake3_dispatch.c"
  "${BLAKE3_DIR}/blake3_portable.c"
)

nanobind_add_module(
  _bcfp
  NB_STATIC
  LTO
  src/bcfp.cpp
  src/bcfp_pybind.cpp
  ${BLAKE3_SOURCES}
)
target_compile_features(_bcfp PRIVATE cxx_std_20)
target_compile_definitions(
  _bcfp
  PRIVATE
    BLAKE3_USE_NEON=0
    BLAKE3_NO_SSE2=1
    BLAKE3_NO_SSE41=1
    BLAKE3_NO_AVX2=1
    BLAKE3_NO_AVX512=1
)
target_compile_options(_bcfp PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
target_include_directories(
  _bcfp
  PRIVATE
    "${RDKIT_INCLUDE_DIR}"
    "${XXHASH_INCLUDE_DIR}"
    "${BLAKE3_DIR}"
)
target_link_libraries(_bcfp PRIVATE ${RDKIT_LIBRARIES})
set_target_properties(
  _bcfp
  PROPERTIES
    BUILD_RPATH "${RDKIT_PREFIX}/lib"
    INSTALL_RPATH "${RDKIT_PREFIX}/lib"
)

install(TARGETS _bcfp LIBRARY DESTINATION . RUNTIME DESTINATION .)

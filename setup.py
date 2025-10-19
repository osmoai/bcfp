#!/usr/bin/env python3
"""
Setup script for BCFP (Bond-Centered Fingerprints & ECFP)
High-performance molecular fingerprints with 3 hash function implementations
"""

from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os
import sys
import sysconfig

def find_rdkit_paths():
    """Auto-detect RDKit installation paths."""
    # Try conda environment
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    if conda_prefix:
        include = os.path.join(conda_prefix, 'include')
        lib = os.path.join(conda_prefix, 'lib')
        if os.path.exists(os.path.join(include, 'rdkit')):
            return include, lib
    
    # Try system Python site-packages
    site_packages = sysconfig.get_paths()["purelib"]
    rdkit_include = os.path.join(site_packages, 'rdkit', 'include')
    if os.path.exists(rdkit_include):
        return rdkit_include, os.path.join(site_packages, 'rdkit', 'lib')
    
    # Fallback to common locations
    common_paths = [
        ('/usr/local/include', '/usr/local/lib'),
        ('/opt/homebrew/include', '/opt/homebrew/lib'),
        ('/usr/include', '/usr/lib'),
    ]
    
    for include_path, lib_path in common_paths:
        if os.path.exists(os.path.join(include_path, 'rdkit')):
            return include_path, lib_path
    
    print("Warning: Could not auto-detect RDKit paths. Using system defaults.")
    return '', ''

rdkit_include, rdkit_lib = find_rdkit_paths()

# Build include directories
include_dirs = []
if rdkit_include:
    include_dirs.extend([rdkit_include, os.path.join(rdkit_include, 'rdkit')])

# System include paths
system_includes = ['/usr/local/include', '/opt/homebrew/include']
include_dirs.extend([p for p in system_includes if os.path.exists(p)])

# Build library directories
library_dirs = []
if rdkit_lib:
    library_dirs.append(rdkit_lib)

system_lib_dirs = ['/usr/local/lib', '/opt/homebrew/lib']
library_dirs.extend([p for p in system_lib_dirs if os.path.exists(p)])

# Platform-specific compilation flags
extra_compile_args = ['-O3', '-ffast-math'] if sys.platform != 'win32' else ['/O2']
if sys.platform == 'darwin':
    extra_compile_args.append('-march=native')
elif sys.platform == 'linux':
    extra_compile_args.append('-march=native')

# Define the C++ extension
ext_modules = [
    Pybind11Extension(
        "_bcfp",
        sources=[
            "src/bcfp.cpp",
            "src/bcfp_pybind.cpp"
        ],
        include_dirs=include_dirs,
        libraries=[
            "RDKitGraphMol",
            "RDKitSmilesParse",
            "RDKitDataStructs",
            "RDKitFingerprints",
            "xxhash",
            "blake3"
        ],
        library_dirs=library_dirs,
        language='c++',
        cxx_std=17,
        extra_compile_args=extra_compile_args,
    ),
]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bcfp",
    version="2.0.0",
    author="BCFP Contributors",
    author_email="",
    description="High-performance molecular fingerprints (ECFP/BCFP) with 3 hash function implementations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bcfp",
    packages=find_packages(exclude=["tests", "tests.*", "benchmarks", "paper", "archive"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "rdkit>=2020.09.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
        "ml": [
            "xgboost>=1.5.0",
            "scikit-learn>=0.24.0",
            "pandas>=1.2.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords="molecular-fingerprints ecfp bcfp cheminformatics hash-functions",
    zip_safe=False,
)


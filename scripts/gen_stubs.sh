#!/bin/bash
set -e

# Ensure we are in the project root or adjust paths
# Usage: ./scripts/gen_stubs.sh

echo "Generating Type Stubs for aeon_py..."

# Install stubgen if not present (requires nanobind package)
# python -m pip install nanobind 

# Run stubgen on the installed/built module
# We treat 'aeon_py.core' as the target.
# Check where the .so is. usually shell/aeon_py/core...so
# We rely on python finding it via PYTHONPATH or installed site-packages.

# Adding shell to PYTHONPATH to find the package locally
# Run against installed package
# export PYTHONPATH=$PYTHONPATH:$(pwd)/shell

# Output to shell/aeon_py/ (so core.pyi sits next to core.cpython...)
python -m nanobind.stubgen --module aeon_py.core -O shell/aeon_py/ --recursive

echo "Stubs generated in shell/aeon_py/"

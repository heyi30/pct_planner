#!/bin/bash

ROOT_DIR=$(cd $(dirname "$0"); pwd)
# echo "ROOT_DIR: ${ROOT_DIR}"

cd lib

# Fall back to a user-specific build directory if the default one is not writable.
BUILD_DIR="./build"
if [ -d "$BUILD_DIR" ] && [ ! -w "$BUILD_DIR" ]; then
    BUILD_DIR="./build_${USER}"
fi

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

cd "$BUILD_DIR"
cmake ../ -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j6
cp ./src/sparse_a_star/sparse_a_star*.so ../
cd ..

# # optional
export PYTHONPATH=$PYTHONPATH:${ROOT_DIR}/lib


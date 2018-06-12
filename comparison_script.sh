#!/bin/bash

git clone git@bitbucket.org:dmaljovec/ngl-beta.git ngl-base
mkdir ngl-base/build
pushd ngl-base/build
cmake ..
make
time binsrc/getNeighborGraph -d 2 -b 1 -k 100 -i ../../data_2_1000000_0.csv > ../../edges_2D_base.txt
popd

git clone git@bitbucket.org:dmaljovec/ngl-beta.git ngl-omp
pushd ngl-omp
git checkout omp && git pull
mkdir build
pushd build
cmake ..
make
time binsrc/getNeighborGraph -d 2 -b 1 -k 100 -i ../../data_2_1000000_0.csv > ../../edges_2D_omp.txt
popd

git clone git@github.com:maljovec/nglpy.git nglpy
pushd nglpy
git checkout pure_python
python test_numba.py -d 2 -n 1000000
popd

cd ngl-gpu
nvcc prune.cu -lann -o prune_graph && time ./prune_graph
popd

python compare_edge_sets.py
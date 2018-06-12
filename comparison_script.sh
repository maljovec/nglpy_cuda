#!/bin/bash

build_dir=$(pwd)/data/misc
input_dir=$(pwd)/data/input
output_dir=$(pwd)/data/output

rm -rf ngl-base ngl-omp nglpy build-ngl-base.txt build-ngl-omp.txt build-ngl-numba.txt build-ngl-cuda.txt

echo "===== Cloning and building ngl-base ====="
start=$(date +%s)
echo "===== Cloning =====" > $build_dir/build-ngl-base.txt 2>&1
git clone git@bitbucket.org:dmaljovec/ngl-beta.git ngl-base >> $build_dir/build-ngl-base.txt 2>&1
mkdir ngl-base/build >> $build_dir/build-ngl-base.txt 2>&1
pushd ngl-base/build >> $build_dir/build-ngl-base.txt 2>&1
echo "===== Building =====" >> $build_dir/build-ngl-base.txt 2>&1
cmake .. >> $build_dir/build-ngl-base.txt 2>&1
make >> $build_dir/build-ngl-base.txt 2>&1
popd > /dev/null
end=$(date +%s)
let "diff=$end-$start"
echo "$diff seconds"

echo "===== Cloning and building ngl-omp ====="
start=$(date +%s)
echo "===== Cloning =====" > $build_dir/build-ngl-omp.txt 2>&1
git clone git@bitbucket.org:dmaljovec/ngl-beta.git ngl-omp >> $build_dir/build-ngl-omp.txt 2>&1
mkdir ngl-omp/build >> $build_dir/build-ngl-omp.txt 2>&1
pushd ngl-omp >> $build_dir/build-ngl-omp.txt 2>&1
git checkout omp >> $build_dir/build-ngl-omp.txt 2>&1
git pull >> $build_dir/build-ngl-omp.txt 2>&1
echo "===== Building =====" >> $build_dir/build-ngl-omp.txt 2>&1
cd build >> $build_dir/build-ngl-omp.txt 2>&1
cmake .. >> $build_dir/build-ngl-omp.txt 2>&1
make >> $build_dir/build-ngl-omp.txt 2>&1
popd > /dev/null
end=$(date +%s)
let "diff=$end-$start"
echo "$diff seconds"

echo "===== Cloning and building ngl-numba ====="
start=$(date +%s)
echo "===== Cloning =====" > $build_dir/build-ngl-numba.txt 2>&1
git clone git@github.com:maljovec/nglpy.git nglpy >> $build_dir/build-ngl-numba.txt 2>&1
pushd nglpy >> $build_dir/build-ngl-numba.txt 2>&1
git checkout pure_python >> $build_dir/build-ngl-numba.txt 2>&1
git pull >> $build_dir/build-ngl-numba.txt 2>&1
popd > /dev/null
end=$(date +%s)
let "diff=$end-$start"
echo "$diff seconds"

echo "===== Building ngl-gpu ====="
start=$(date +%s)
echo "===== Building =====" > $build_dir/build-ngl-cuda.txt 2>&1
pushd ngl-gpu >> $build_dir/build-ngl-cuda.txt 2>&1
nvcc prune.cu -lann -o prune_graph >> $build_dir/build-ngl-cuda.txt 2>&1
popd > /dev/null
end=$(date +%s)
let "diff=$end-$start"
echo "$diff seconds"

pushd ngl-base/build > /dev/null
echo "-----ngl-base------------------------------------------------------------"
time binsrc/getNeighborGraph -d 2 -b 1 -k 100 -i $input_dir/data_2_1000000_0.csv > $output_dir/edges_2D_base.txt
echo "-------------------------------------------------------------------------"
popd > /dev/null

pushd ngl-omp/build > /dev/null
echo "-----ngl-omp-------------------------------------------------------------"
time binsrc/getNeighborGraph -d 2 -b 1 -k 100 -i $input_dir/data_2_1000000_0.csv > $output_dir/edges_2D_omp.txt
echo "-------------------------------------------------------------------------"
popd > /dev/null

pushd nglpy > /dev/null
echo "-----ngl-numba-----------------------------------------------------------"
time python test_numba.py -i $input_dir/data_2_1000000_0.csv > $output_dir/edges_2D_numba.txt
echo "-------------------------------------------------------------------------"
popd > /dev/null

pushd ngl-gpu > /dev/null
nvcc prune.cu -lann -o prune_graph
echo "-----ngl-cuda-discrete---------------------------------------------------"
time ./prune_graph -i $input_dir/data_2_1000000_0.csv -d 2 -n 1000000 -k 100 -b 1 -p 2 -s 9999 > $output_dir/edges_2D_gpu_discrete.txt
echo "-----ngl-cuda------------------------------------------------------------"
time ./prune_graph -i $input_dir/data_2_1000000_0.csv -d 2 -n 1000000 -k 100 -b 1 -p 2 -s -1 > $output_dir/edges_2D_gpu.txt
echo "-------------------------------------------------------------------------"
popd > /dev/null

python compare_edge_sets.py 2
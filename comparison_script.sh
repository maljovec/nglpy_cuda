#!/bin/bash

build_dir=$(pwd)/data/misc
input_dir=$(pwd)/data/input
output_dir=$(pwd)/data/output

D=2
N=1000000
K=100

echo "D=$D N=$N K=$K"

build_file=$build_dir/build-ngl-

rm -rf $build_dir/build-ngl-base.txt $build_dir/build-ngl-omp.txt $build_dir/build-ngl-numba.txt $build_dir/build-ngl-cuda.txt
rm -rf $output_dir/edges_${D}D_base.txt $output_dir/edges_${D}D_omp.txt $output_dir/edges_${D}D_numba.txt $output_dir/edges_${D}D_gpu.txt $output_dir/edges_${D}D_gpu_discrete.txt

################################################################################
# TODO: Add check for "hard reset" which will perform this nuke, otherwise use
# the existing directories to pull and build.
# if [ 0 ]
# then
#   rm -rf ngl-base ngl-omp ngl-numba 
# fi
################################################################################
echo "===== Updating and building ngl-base ====="
start=$(date +%s)
if [ ! -d ngl-base ]
then
    echo "===== Cloning =====" &> $build_dir/build-ngl-base.txt
    git clone git@bitbucket.org:dmaljovec/ngl-beta.git ngl-base &>> $build_dir/build-ngl-base.txt
fi
echo "===== Updating =====" &>> $build_dir/build-ngl-base.txt
pushd ngl-base &>> $build_dir/build-ngl-base.txt
git pull &>> $build_dir/build-ngl-base.txt
echo "===== Building =====" &>> $build_dir/build-ngl-base.txt
cmake . &>> $build_dir/build-ngl-base.txt
make &>> $build_dir/build-ngl-base.txt
popd > /dev/null
end=$(date +%s)
let "diff=$end-$start"
echo "$diff seconds"

echo "===== Updating and building ngl-omp ====="
start=$(date +%s)
if [ ! -d ngl-omp ]
then
    echo "===== Cloning =====" &> $build_dir/build-ngl-omp.txt
    git clone git@bitbucket.org:dmaljovec/ngl-beta.git ngl-omp &>> $build_dir/build-ngl-omp.txt
fi
echo "===== Updating =====" &>> $build_dir/build-ngl-omp.txt
pushd ngl-omp &>> $build_dir/build-ngl-omp.txt
git checkout omp &>> $build_dir/build-ngl-omp.txt
git pull &>> $build_dir/build-ngl-omp.txt
echo "===== Building =====" &>> $build_dir/build-ngl-omp.txt
cmake . &>> $build_dir/build-ngl-omp.txt
make &>> $build_dir/build-ngl-omp.txt
popd > /dev/null
end=$(date +%s)
let "diff=$end-$start"
echo "$diff seconds"

# echo "===== Updating and building ngl-numba ====="
# start=$(date +%s)
# if [ ! -d ngl-numba ]
# then
#     echo "===== Cloning =====" &> $build_dir/build-ngl-numba.txt
#     git clone git@github.com:maljovec/nglpy.git ngl-numba &>> $build_dir/build-ngl-numba.txt
# fi
# echo "===== Updating =====" &>> $build_dir/build-ngl-numba.txt
# pushd ngl-numba &>> $build_dir/build-ngl-numba.txt
# git checkout pure_python &>> $build_dir/build-ngl-numba.txt
# git pull &>> $build_dir/build-ngl-numba.txt
# popd > /dev/null
# end=$(date +%s)
# let "diff=$end-$start"
# echo "$diff seconds"

echo "===== Building ngl-gpu ====="
start=$(date +%s)
echo "===== Building =====" > $build_dir/build-ngl-cuda.txt
pushd ngl-gpu &>> $build_dir/build-ngl-cuda.txt
nvcc prune.cu -lann -o prune_graph &>> $build_dir/build-ngl-cuda.txt
popd > /dev/null
end=$(date +%s)
let "diff=$end-$start"
echo "$diff seconds"

pushd ngl-base > /dev/null
echo "-----ngl-base------------------------------------------------------------"
time binsrc/getNeighborGraph -d $D -b 1 -k $K -i $input_dir/data_${D}_${N}_0.csv > $output_dir/edges_${D}D_base.txt
echo "-------------------------------------------------------------------------"
popd > /dev/null

pushd ngl-omp > /dev/null
echo "-----ngl-omp-------------------------------------------------------------"
time binsrc/getNeighborGraph -d $D -b 1 -k $K -i $input_dir/data_${D}_${N}_0.csv > $output_dir/edges_${D}D_omp.txt
echo "-------------------------------------------------------------------------"
popd > /dev/null

# pushd ngl-numba > /dev/null
# echo "-----ngl-numba-----------------------------------------------------------"
# time python test_numba.py -i $input_dir/data_${D}_${N}_0.csv -s 999 > $output_dir/edges_${D}D_numba.txt
# echo "-------------------------------------------------------------------------"
# popd > /dev/null

pushd ngl-gpu > /dev/null
echo "-----ngl-cuda-discrete---------------------------------------------------"
time ./prune_graph -i $input_dir/data_${D}_${N}_0.csv -d $D -n $N -k $K -b 1 -p 2 -s 9999 > $output_dir/edges_${D}D_gpu_discrete.txt
echo "-------------------------------------------------------------------------"

echo "-----ngl-cuda------------------------------------------------------------"
time ./prune_graph -i $input_dir/data_${D}_${N}_0.csv -d $D -n $N -k $K -b 1 -p 2 -s -1 > $output_dir/edges_${D}D_gpu.txt
echo "-------------------------------------------------------------------------"
popd > /dev/null

python compare_edge_sets.py $D
#!/bin/bash

build_dir=$(pwd)/data/misc
input_dir=$(pwd)/data/input
output_dir=$(pwd)/data/output

D=3
N=1000000
K=100

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -c|--count)
    N="$2"
    shift # past argument
    shift # past value
    ;;
    -d|--dimensionality)
    D="$2"
    shift # past argument
    shift # past value
    ;;
    -k|--kneighbors)
    K="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

echo "D=$D N=$N K=$K"

build_file=$build_dir/build-ngl-
edge_file=$output_dir/edges_${D}D_

rm -rf ${build_file}base.txt ${build_file}omp.txt ${build_file}numba.txt ${build_file}cuda.txt
rm -rf ${edge_file}base.txt ${edge_file}omp.txt ${edge_file}numba.txt ${edge_file}gpu.txt ${edge_file}gpu_discrete.txt

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
    echo "===== Cloning =====" &> ${build_file}base.txt
    git clone git@bitbucket.org:dmaljovec/ngl-beta.git ngl-base &>> ${build_file}base.txt
fi
echo "===== Updating =====" &>> ${build_file}base.txt
pushd ngl-base &>> ${build_file}base.txt
git pull &>> ${build_file}base.txt
echo "===== Building =====" &>> ${build_file}base.txt
cmake . &>> ${build_file}base.txt
make &>> ${build_file}base.txt
popd > /dev/null
end=$(date +%s)
let "diff=$end-$start"
echo "$diff seconds"

echo "===== Updating and building ngl-omp ====="
start=$(date +%s)
if [ ! -d ngl-omp ]
then
    echo "===== Cloning =====" &> ${build_file}omp.txt
    git clone git@bitbucket.org:dmaljovec/ngl-beta.git ngl-omp &>> ${build_file}omp.txt
fi
echo "===== Updating =====" &>> ${build_file}omp.txt
pushd ngl-omp &>> ${build_file}omp.txt
git checkout omp &>> ${build_file}omp.txt
git pull &>> ${build_file}omp.txt
echo "===== Building =====" &>> ${build_file}omp.txt
cmake . &>> ${build_file}omp.txt
make &>> ${build_file}omp.txt
popd > /dev/null
end=$(date +%s)
let "diff=$end-$start"
echo "$diff seconds"

# echo "===== Updating and building ngl-numba ====="
# start=$(date +%s)
# if [ ! -d ngl-numba ]
# then
#     echo "===== Cloning =====" &> ${build_file}numba.txt
#     git clone git@github.com:maljovec/nglpy.git ngl-numba &>> ${build_file}numba.txt
# fi
# echo "===== Updating =====" &>> ${build_file}numba.txt
# pushd ngl-numba &>> ${build_file}numba.txt
# git checkout pure_python &>> ${build_file}numba.txt
# git pull &>> ${build_file}numba.txt
# popd > /dev/null
# end=$(date +%s)
# let "diff=$end-$start"
# echo "$diff seconds"

echo "===== Building ngl-gpu ====="
start=$(date +%s)
echo "===== Building =====" > ${build_file}cuda.txt
pushd ngl-gpu &>> ${build_file}cuda.txt
nvcc prune.cu -lann -o prune_graph &>> ${build_file}cuda.txt
popd > /dev/null
end=$(date +%s)
let "diff=$end-$start"
echo "$diff seconds"

pushd ngl-base > /dev/null
echo "-----ngl-base------------------------------------------------------------"
time binsrc/getNeighborGraph -d $D -b 1 -k $K -i $input_dir/data_${D}_${N}_0.csv > ${edge_file}base.txt
echo "-------------------------------------------------------------------------"
popd > /dev/null

pushd ngl-omp > /dev/null
echo "-----ngl-omp-------------------------------------------------------------"
time binsrc/getNeighborGraph -d $D -b 1 -k $K -i $input_dir/data_${D}_${N}_0.csv > ${edge_file}omp.txt
echo "-------------------------------------------------------------------------"
popd > /dev/null

# pushd ngl-numba > /dev/null
# echo "-----ngl-numba-----------------------------------------------------------"
# time python test_numba.py -i $input_dir/data_${D}_${N}_0.csv -s 999 > ${edge_file}numba.txt
# echo "-------------------------------------------------------------------------"
# popd > /dev/null

pushd ngl-gpu > /dev/null
echo "-----ngl-cuda-discrete---------------------------------------------------"
time ./prune_graph -d $D -b 1 -k $K -i $input_dir/data_${D}_${N}_0.csv -n $N -p 2 -s 9999 > ${edge_file}gpu_discrete.txt
echo "-------------------------------------------------------------------------"

echo "-----ngl-cuda------------------------------------------------------------"
time ./prune_graph -d $D -b 1 -k $K -i $input_dir/data_${D}_${N}_0.csv -n $N -p 2 -s -1 > ${edge_file}gpu.txt
echo "-------------------------------------------------------------------------"
popd > /dev/null

python compare_edge_sets.py $D
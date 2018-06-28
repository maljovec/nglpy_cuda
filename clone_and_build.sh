#!/bin/bash

build_dir=$(pwd)/data/misc
build_file=$build_dir/build-ngl-
rm -rf ${build_file}base.txt ${build_file}omp.txt ${build_file}numba.txt ${build_file}cuda.txt

################################################################################
# TODO: Add check for "hard reset" which will perform this nuke, otherwise use
# the existing directories to pull and build.
# if [ 0 ]
# then
#   rm -rf ngl-base ngl-omp ngl-numba
# fi
################################################################################
echo "===== Updating and building knn-generator ====="
start=$(date +%s)
if [ ! -d knn-generator ]
then
    echo "===== Cloning =====" &> ${build_file}knn.txt
    git clone git@bitbucket.org:dmaljovec/ngl-beta.git knn-generator &>> ${build_file}knn.txt
fi
echo "===== Updating =====" &>> ${build_file}knn.txt
pushd knn-generator &>> ${build_file}knn.txt
git checkout knn_generator &>> ${build_file}knn.txt
git pull &>> ${build_file}knn.txt
echo "===== Building =====" &>> ${build_file}knn.txt
cmake . &>> ${build_file}knn.txt
make &>> ${build_file}knn.txt
popd > /dev/null
end=$(date +%s)
let "diff=$end-$start"
echo "$diff seconds"

echo "===== Updating and building cvt ====="
start=$(date +%s)
if [ ! -d cvt ]
then
    echo "===== Cloning =====" &> ${build_file}cvt.txt
    git clone git@bitbucket.org:dmaljovec/cvt.git cvt &>> ${build_file}cvt.txt
fi
echo "===== Updating =====" &>> ${build_file}cvt.txt
pushd cvt &>> ${build_file}cvt.txt
git pull &>> ${build_file}cvt.txt
echo "===== Building =====" &>> ${build_file}cvt.txt
make &>> ${build_file}cvt.txt
popd > /dev/null
end=$(date +%s)
let "diff=$end-$start"
echo "$diff seconds"

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

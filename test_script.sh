nvcc src/ngl_cuda.cu src/ANNSearchIndex.cpp src/Graph.cpp -I include/ --compiler-options "-fPIC" --shared -o libnglcu.so -lann
g++ -L. -I include/ src/testAPI.cpp -lnglcu -o test

echo "########## Strict, Continuous  ##########"
./test -d 2 -c 10 -k 10 -b 1 -p 2 -r 0
# echo "########## Relaxed, Continuous ##########"
# ./test -d 2 -c 10 -k 10 -b 1 -p 2 -r 1
# echo "##########  Strict, Discrete   ##########"
# ./test -d 2 -c 10 -k 10 -b 1 -p 2 -r 0 -s 100
# echo "##########  Relaxed, Discrete  ##########"
# ./test -d 2 -c 10 -k 10 -b 1 -p 2 -r 1 -s 100

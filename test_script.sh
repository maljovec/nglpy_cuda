nvcc src/ngl_cuda.cu -I include/ --compiler-options "-fPIC" --shared -o libnglcu.so
g++ -L. -I include/ src/test.cpp -lnglcu -o test

echo "########## Strict, Continuous  ##########"
./test -i nglpy_cuda/tests/data/points.txt -d 2 -c 10 -n nglpy_cuda/tests/data/edges.txt -k 10 -b 1 -p 2 -r 0
echo "########## Relaxed, Continuous ##########"
./test -i nglpy_cuda/tests/data/points.txt -d 2 -c 10 -n nglpy_cuda/tests/data/edges.txt -k 10 -b 1 -p 2 -r 1
echo "##########  Strict, Discrete   ##########"
./test -i nglpy_cuda/tests/data/points.txt -d 2 -c 10 -n nglpy_cuda/tests/data/edges.txt -k 10 -b 1 -p 2 -r 0 -s 100
echo "##########  Relaxed, Discrete  ##########"
./test -i nglpy_cuda/tests/data/points.txt -d 2 -c 10 -n nglpy_cuda/tests/data/edges.txt -k 10 -b 1 -p 2 -r 1 -s 100
# ./test -i nglpy_cuda/tests/data/points.txt -d 2 -c 10 -n nglpy_cuda/tests/data/edges.txt -k 10 -b 1 -p 2 -r 0
# ./test -i nglpy_cuda/tests/data/points.txt -d 2 -c 10 -n nglpy_cuda/tests/data/edges.txt -k 10 -b 1 -p 2 -r 1
# ./test -i nglpy_cuda/tests/data/points.txt -d 2 -c 10 -n nglpy_cuda/tests/data/edges.txt -k 10 -b 1 -p 2 -s 100 -r 0
# ./test -i nglpy_cuda/tests/data/points.txt -d 2 -c 10 -n nglpy_cuda/tests/data/edges.txt -k 10 -b 1 -p 2 -s 100 -r 1

nvcc src/ngl_cuda.cu -I include/ --compiler-options "-fPIC" --shared -o libnglcu.so
g++ -L. -I include/ src/test.cpp -lnglcu -o test
./test -i data/points.txt -d 2 -c 10 -n data/edges.txt -k 10 -b 1 -p 2
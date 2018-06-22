#!/bin/bash
time ./comparison_script.sh -c 1000000 -d 4 -k 100 &> ../4D_results.txt
time ./comparison_script.sh -c 1000000 -d 5 -k 200 &> ../5D_results.txt
time ./comparison_script.sh -c 1000000 -d 6 -k 200 &> ../6D_results.txt
time ./comparison_script.sh -c 1000000 -d 7 -k 300 &> ../7D_results.txt

time ../knn-generator/binsrc/getNeighborGraph -d 8 -b 1 -k 300 -i ../ngl-gpu/data/input/data_8_1000000_0.csv 2> data/misc/knn_8D.txt

time ./comparison_script.sh -c 1000000 -d 8 -k 300 &> ../8D_results.txt

time ../knn-generator/binsrc/getNeighborGraph -d 9 -b 1 -k 300 -i ../ngl-gpu/data/input/data_9_1000000_0.csv 2> data/misc/knn_9D.txt

time ./comparison_script.sh -c 1000000 -d 9 -k 300 &> ../9D_results.txt

time ../knn-generator/binsrc/getNeighborGraph -d 10 -b 1 -k 300 -i ../ngl-gpu/data/input/data_10_1000000_0.csv 2> data/misc/knn_10D.txt

time ./comparison_script.sh -c 1000000 -d 10 -k 300 &> ../10D_results.txt

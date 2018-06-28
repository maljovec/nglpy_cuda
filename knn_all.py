import subprocess
import os
import sys
import time
# python test_all.py > test.out


algorithms = ['base', 'omp', 'gpu', 'gpud']

print('Dimension, Seed, Sampling, Point_Count, #_Neighbors, Algorithm, Time, Total_Edge_Count, Missed_Edge_Count, Additional_Edge_Count')
for N in [1000000]:
    for D in range(7, 10):
        for S in range(0, 10):
            for A in ['uniform', 'normal', 'lhs', 'cvt']:
                if D < 5:
                    K = 100
                elif D < 7:
                    K = 200
                elif D < 9:
                    K = 300
                else:
                    K = 500

                K = min(K, N)
                p_filename = 'data/input/points_{}_{}_{}_{}.csv'.format(A, N, D, S)
                k_filename = 'data/graphs/knn_{}_{}_{}_{}.txt'.format(A, N, D, S)

                # Generate Point Set
                if not os.path.exists(p_filename) or not os.stat(p_filename).st_size:
                    sys.stderr.write(p_filename + ' generation: ')
                    sys.stderr.flush()
                    start = time.time()
                    result = subprocess.run(['python', 'generate_sample_set.py', str(D), str(S), str(N), str(A), p_filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                    end = time.time()
                    sys.stderr.write('{} s\n'.format(end-start))

                # Generate Graph to Prune
                if not os.path.exists(k_filename) or not os.stat(k_filename).st_size:
                    sys.stderr.write(k_filename + ' generation: ')
                    sys.stderr.flush()
                    start = time.time()
                    fptr = open(k_filename, 'w')
                    result = subprocess.run(['../knn-generator/binsrc/getNeighborGraph', '-d', str(D), '-b', '1', '-k', str(K), '-i', p_filename], stdout=subprocess.PIPE, stderr=fptr, check=True)
                    fptr.close()
                    end = time.time()
                    sys.stderr.write('{} s\n'.format(end-start))

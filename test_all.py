import subprocess
import os
# python generate_all_samples.py > test.out

algorithms = ['base', 'omp', 'gpu', 'gpud']

print('Dimension, Seed, Sampling, Point_Count, #_Neighbors, Algorithm, Time, Total_Edge_Count, Missed_Edge_Count, Additional_Edge_Count')
for N in [10000, 100000, 1000000]:
    for D in range(2, 8):
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
                result = subprocess.run(['python', 'generate_sample_set.py', str(D), str(S), str(N), str(A), p_filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)            

                # Generate Graph to Prune
                fptr = open(k_filename, 'w')
                result = subprocess.run(['../knn-generator/binsrc/getNeighborGraph', '-d', str(D), '-b', '1', '-k', str(K), '-i', p_filename], stdout=subprocess.PIPE, stderr=fptr, check=True)
                fptr.close()

                # Initialize dictionaries for storing the commands for each
                # algorithm, the execution times, and the files to store the pruned
                # edge lists
                cmds = {}
                times = {}
                filenames = {}
                for alg in algorithms:
                    filenames[alg] = 'data/output/{}_{}_{}_{}_{}.txt'.format(alg, A, N, D, S)

                cmds['base'] = ['/usr/bin/time','-f','%e', 'ngl-base/binsrc/getNeighborGraph', '-d', str(D), '-b', '1', '-k', str(K), '-i', p_filename, '-n', k_filename]
                cmds['omp'] = ['/usr/bin/time','-f','%e', 'ngl-omp/binsrc/getNeighborGraph', '-d', str(D), '-b', '1', '-k', str(K), '-i', p_filename, '-n', k_filename]
                cmds['gpud'] = ['/usr/bin/time','-f','%e', 'ngl-gpu/prune_graph', '-d', str(D), '-b', '1',  '-k', str(K), '-i', p_filename, '-n', k_filename, '-c', str(N), '-p', '2', '-s', '9999']
                cmds['gpu'] = ['/usr/bin/time','-f','%e', 'ngl-gpu/prune_graph', '-d', str(D), '-b', '1',  '-k', str(K), '-i', p_filename, '-n', k_filename, '-c', str(N), '-p', '2', '-s', '-1']

                # Run the algorithms
                for alg in algorithms:
                    fptr = open(filenames[alg], 'w')
                    result = subprocess.run(cmds[alg], stdout=fptr, stderr=subprocess.PIPE, check=True)
                    fptr.close()
                    times[alg] = float(result.stderr.decode('utf-8').strip().split('\n')[-1])

                # Count the total number of edges and the differences between
                # the base algorithm and the others
                counts = {}
                base_only = {}
                other_only = {}
                for alg,fname in filenames.items():
                    if os.path.isfile(fname):
                        eset = set()

                        fptr = open(fname)
                        for line in fptr:
                            edge = list(map(int, line.split(' ')))
                            if edge[1] < edge[0]:
                                lo = edge[1]
                                hi = edge[0]
                            else:
                                lo = edge[0]
                                hi = edge[1]
                            
                            if lo != hi:
                                eset.add((lo,hi))
                        
                        counts[alg] = len(eset)

                        if alg == 'base':
                            eset_base = set(eset)
                            base_only[alg] = 0
                            other_only[alg] = 0
                        else:
                            base_only[alg] = len(eset_base.difference(eset))
                            other_only[alg] = len(eset.difference(eset_base))

                for alg in algorithms:
                    print('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(D, S, A, N, K, alg, times[alg], counts[alg], base_only[alg], other_only[alg]))
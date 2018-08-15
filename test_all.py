import subprocess
import os
import sys
import time
# python test_all.py > test.out

knn_program = 'knn-generator/binsrc/getNeighborGraph'
algorithms = ['base', 'omp', 'numba']  # , 'gpu', 'gpud']

print('Dimension, Seed, Sampling, Point_Count, #_Neighbors, Algorithm, Time, '
      'Total_Edge_Count, Missed_Edge_Count, Additional_Edge_Count')
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
                p_file = 'data/input/points_{}_{}_{}_{}.csv'.format(A, N, D, S)
                k_file = 'data/graphs/knn_{}_{}_{}_{}.txt'.format(A, N, D, S)

                # Generate Point Set
                if not os.path.exists(p_file) or not os.stat(p_file).st_size:
                    sys.stderr.write(p_file + ' generation: ')
                    sys.stderr.flush()
                    start = time.time()
                    result = subprocess.run(['python',
                                             'generate_sample_set.py',
                                             str(D), str(S), str(N), str(A),
                                             p_file],
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, check=True)
                    end = time.time()
                    sys.stderr.write('{} s\n'.format(end-start))

                # Generate Graph to Prune
                if not os.path.exists(k_file) or not os.stat(k_file).st_size:
                    sys.stderr.write(k_file + ' generation: ')
                    sys.stderr.flush()
                    start = time.time()
                    fptr = open(k_file, 'w')
                    result = subprocess.run([knn_program,
                                             '-d', str(D), '-b', '1', '-k',
                                             str(K), '-i', p_file],
                                            stdout=subprocess.PIPE,
                                            stderr=fptr, check=True)
                    fptr.close()
                    end = time.time()
                    sys.stderr.write('{} s\n'.format(end-start))

                # Initialize dictionaries for storing the commands for each
                # algorithm, the execution times, and the files to store the
                # pruned edge lists
                cmds = {}
                times = {}
                filenames = {}
                for alg in algorithms:
                    filenames[alg] = 'data/output/{}_{}_{}_{}_{}.txt'.format(
                        alg, A, N, D, S)

                cmds['base'] = ['/usr/bin/time', '-f', '%e',
                                'ngl-base/binsrc/getNeighborGraph', '-d',
                                str(D), '-b', '1', '-k', str(K), '-i',
                                p_file, '-n', k_file]
                cmds['omp'] = ['/usr/bin/time', '-f', '%e',
                               'ngl-omp/binsrc/getNeighborGraph', '-d',
                               str(D), '-b', '1', '-k', str(K), '-i',
                               p_file, '-n', k_file]
                cmds['numba'] = ['/usr/bin/time', '-f', '%e', 'python',
                                 'ngl-numba/test_numpy.py', '-b', '1', '-i',
                                 p_file, '-n', k_file, '-s', '9999',
                                 '-p', '2']
                cmds['gpud'] = ['/usr/bin/time', '-f', '%e',
                                'ngl-gpu/prune_graph', '-d', str(D), '-b', '1',
                                '-k', str(K), '-i', p_file, '-n',
                                k_file, '-c', str(N), '-p', '2', '-s',
                                '9999']
                cmds['gpu'] = ['/usr/bin/time', '-f', '%e',
                               'ngl-gpu/prune_graph', '-d', str(D), '-b', '1',
                               '-k', str(K), '-i', p_file, '-n',
                               k_file, '-c', str(N), '-p', '2', '-s', '-1']

                # Run the algorithms
                for alg in algorithms:
                    fptr = open(filenames[alg], 'w')
                    sys.stderr.write(filenames[alg] + ' computation: ')
                    sys.stderr.flush()
                    start = time.time()
                    result = subprocess.run(cmds[alg], stdout=fptr,
                                            stderr=subprocess.PIPE, check=True)
                    end = time.time()
                    sys.stderr.write('{} s\n'.format(end-start))
                    fptr.close()
                    times[alg] = float(result.stderr.decode(
                        'utf-8').strip().split('\n')[-1])

                os.remove(p_file)
                # os.remove(k_file)
                # Count the total number of edges and the differences between
                # the base algorithm and the others
                counts = {}
                base_only = {}
                other_only = {}
                for alg, fname in filenames.items():
                    sys.stderr.write(fname + ' comparison')
                    sys.stderr.flush()
                    start = time.time()
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
                                eset.add((lo, hi))

                        counts[alg] = len(eset)

                        if alg == 'base':
                            eset_base = set(eset)
                            base_only[alg] = 0
                            other_only[alg] = 0
                        else:
                            # base_only[alg] = len(eset_base.difference(eset))
                            # other_only[alg] = len(eset.difference(eset_base))
                            base_only[alg] = 'na'
                            other_only[alg] = 'na'
                    end = time.time()
                    sys.stderr.write('{} s\n'.format(end-start))

                for alg in algorithms:
                    print('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(
                        D, S, A, N, K, alg, times[alg], counts[alg],
                        base_only[alg], other_only[alg]))
                    sys.stdout.flush()

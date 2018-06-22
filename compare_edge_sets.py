import numpy as np
import argparse 
import os.path

parser = argparse.ArgumentParser(description='Build an lp-beta skeleton using numba.')
parser.add_argument('dimensionality', type=int,
                    help='The dimensionality of the problem to handle.')

args = parser.parse_args()
D = args.dimensionality

algorithms = ['base',
              'omp',
              'numba',
              'gpu',
              'gpu_discrete'
              ]

for alg in algorithms:
    fname = 'data/output/edges_{}D_{}.txt'.format(D, alg)
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
        
        print('{:>12}: {}'.format(alg, len(eset)))

        if alg == 'base':
            eset_base = set(eset)
        else:
            v1 = len(eset_base.difference(eset))
            v2 = len(eset.difference(eset_base))
            if v1+v2 > 0:
                print('Difference base vs {0}:\n         base only {1}\n {0:>12} only {2}'.format(alg, v1, v2))
            else: 
                print('No Difference base vs {0}'.format(alg))
    else:
        print('No data for {}'.format(alg))
    print('~ '*40)

import numpy as np
import argparse 

parser = argparse.ArgumentParser(description='Build an lp-beta skeleton using numba.')
parser.add_argument('dimensionality', type=int,
                    help='The dimensionality of the problem to handle.')

args = parser.parse_args()
D = args.dimensionality

algorithms = ['base','omp','numba','gpu', 'gpu_discrete']

edges = {}
eset = {}
for alg in algorithms:
    edges[alg] = np.loadtxt('data/output/edges_{}D_{}.txt'.format(D, alg), dtype=int)
    eset[alg] = set()
    
    for edge in edges[alg]:
        if edge[1] < edge[0]:
            lo = edge[1]
            hi = edge[0]
        else:
            lo = edge[0]
            hi = edge[1]
        
        if lo != hi:
            eset[alg].add((lo,hi))
    
    print('{:>12}: {}'.format(alg, len(eset[alg])))

eset_base = eset['base']
for alg in algorithms[1:]:
    eset_test = eset[alg]
    v1 = len(eset_base.difference(eset_test))
    v2 = len(eset_test.difference(eset_base))
    print('Difference base vs {0}:\n         base only {1}\n {0:>12} only {2}'.format(alg, v1, v2))

# print('='*80)
# print(eset['numba'].difference(eset_base))
# print('='*80)
# print(eset_base.difference(eset['numba']))

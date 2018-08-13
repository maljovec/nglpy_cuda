import nglpy_cuda
import numpy as np

assert 0.5 == nglpy_cuda.min_distance_from_edge(0, 1, 2)
assert 0 == nglpy_cuda.min_distance_from_edge(1, 1, 2)

D=2
N=1000
A='uniform'
S=0
k=5
beta = 1
p = 2
steps = 50
p_file = '../ngl-gpu/data/input/points_{}_{}_{}_{}.csv'.format(A, N, D, S)
k_file = '../ngl-gpu/data/graphs/knn_{}_{}_{}_{}_{}.txt'.format(A, N, D, k, S)
X = np.loadtxt(p_file, dtype=np.float32)
edges = np.loadtxt(k_file, dtype=np.int32)

template = nglpy_cuda.create_template(beta, p, steps)
assert len(template) == 50
for i in range(len(template)-1):
    assert template[i] > template[i+1]

print('python:')
print(edges[:5,:])
print('C++:')

ngl_edges = nglpy_cuda.prune(N, D, k, p, beta, X, edges)
print(ngl_edges)
#nglpy_cuda.prune_discrete(N, D, k, steps, beta, p, X, edges)
#nglpy_cuda.prune_discrete()
# nglpy_cuda.get_edge_list()

import numpy as np
import nglpy_cuda as ngl
import sklearn

########################################################################
# Example using the Graph class as an iterator
np.random.seed(2)
X = np.random.uniform(size=(10, 2))
graph = ngl.Graph(X, relaxed=False)

print('Iterating on edges:')
for edge in graph:
    print(edge)

########################################################################
# Example of using the directly exposed GPU methods, note you will have
# to decide what to do when the data does not fit in memory, I make an
# attempt to do it for you above, also make sure to use 32-bit floats
# and ints when you call the CUDA methods.
f32 = np.float32
i32 = np.int32

X = np.array(X, dtype=f32)
knnAlgorithm = sklearn.neighbors.NearestNeighbors(10)
knnAlgorithm.fit(X)
edge_matrix = np.array(knnAlgorithm.kneighbors(X, return_distance=False), dtype=i32)

print('Input')
print(edge_matrix)

edges = np.copy(edge_matrix)
edges_out = ngl.prune(X, edges, 2, 1, False)
print('Gabriel Graph')
print(edges_out)

edges = np.copy(edge_matrix)
edges_relaxed_out = ngl.prune(X, edges, True, 2, 1)
print('Relaxed Gabriel Graph (5 and 7 should be connected)')
print(edges_relaxed_out)

template = np.array(ngl.create_template(1, 2, 100), dtype=f32)

edges = np.copy(edge_matrix)
edges_discrete_out = ngl.prune_discrete(X, edges, template, relaxed=False)
print('Discrete Gabriel Graph (should be the same as the normal version')
print(edges_discrete_out)

edges = np.copy(edge_matrix)
edges_discrete_out = ngl.prune_discrete(X, edges, template, relaxed=True)
print('Discrete Gabriel Graph (5 and 7 should be connected')
print(edges_discrete_out)


edges = np.copy(edge_matrix)
edges_discrete_out = ngl.prune_discrete(X, edges, steps=100, relaxed=False, beta=1, lp=2)
print('Discrete Gabriel Graph (should be the same as the normal version')
print(edges_discrete_out)

edges = np.copy(edge_matrix)
edges_discrete_out = ngl.prune_discrete(X, edges, steps=100, relaxed=True, beta=1, lp=2)
print('Discrete Gabriel Graph (5 and 7 should be connected')
print(edges_discrete_out)

import numpy as np
import nglpy_cuda as ngl
import sklearn

f32 = np.float32
i32 = np.int32

np.random.seed(2)
X = np.random.uniform(size=(10, 2))
graph = ngl.Graph(X, relaxed=False)

print('Iterating on edges:')
for edge in graph:
    print(edge)

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

# edges = np.copy(edge_matrix)
# edges_discrete_out = ngl.prune_discrete(X, edges, False, 100, 1, 2)
# print('Discrete Gabriel Graph (should be the same as the normal version')
# print(edges_discrete_out)

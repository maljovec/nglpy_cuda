import numpy as np
from sklearn import neighbors
import samply


def yao_graph(X, D, k, indices):
    vectors = samply.SCVTSampler.generate_samples(k, D)
    indices_out = -1*np.ones(indices.shape)

    for i in range(X.shape[0]):
        projections = np.dot(X[indices[i]] - X[i], vectors.T)
        representatives = np.argmax(projections, axis=1)
        used = np.zeros(len(vectors), dtype=bool)
        j = 0
        while not np.all(used) and j < indices.shape[1]:
            if indices[i, j] != i:
                rep = representatives[j]
                if not used[rep]:
                    indices_out[i, j] = indices[i, j]
                    used[rep] = True
            j += 1
    return indices_out


def theta_graph(X, D, k, indices):
    vectors = samply.SCVTSampler.generate_samples(k, D)
    indices_out = -1*np.ones(indices.shape)

    for i in range(X.shape[0]):
        projections = np.dot(X[indices[i]] - X[i], vectors.T)
        representatives = np.argmax(projections, axis=1)
        max_projections = np.max(projections, axis=1)
        used = np.zeros(len(vectors), dtype=bool)
        order = reversed(np.argsort(max_projections))
        for j in order:
            if indices[i, j] != i:
                rep = representatives[j]
                if not used[rep]:
                    indices_out[i, j] = indices[i, j]
                    used[rep] = True
                    if np.all(used):
                        break
    return indices_out

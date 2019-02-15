import numpy as np
from sklearn import neighbors
import samply
from .utils import f32, i32


def yao_graph(X, num_sectors, points_per_sector, indices):
    D = X.shape[1]
    vectors = samply.SCVTSampler.generate_samples(num_sectors, D)
    indices_out = -1*np.ones(indices.shape, dtype=i32)

    for i in range(X.shape[0]):
        projections = np.dot(X[indices[i]] - X[i], vectors.T)
        representatives = np.argmax(projections, axis=1)
        vacancies = points_per_sector*np.ones(len(vectors), dtype=bool)
        j = 0
        while np.any(vacancies) and j < indices.shape[1]:
            if indices[i, j] != i:
                rep = representatives[j]
                if vacancies[rep] > 0:
                    indices_out[i, j] = indices[i, j]
                    vacancies[rep] -= 1
            j += 1
    return indices_out


def theta_graph(X, num_sectors, points_per_sector, indices):
    D = X.shape[1]
    vectors = samply.SCVTSampler.generate_samples(num_sectors, D)
    indices_out = -1*np.ones(indices.shape, dtype=i32)

    for i in range(X.shape[0]):
        projections = np.dot(X[indices[i]] - X[i], vectors.T)
        representatives = np.argmax(projections, axis=1)
        max_projections = np.max(projections, axis=1)
        vacancies = points_per_sector*np.ones(len(vectors), dtype=bool)
        order = reversed(np.argsort(max_projections))
        for j in order:
            if indices[i, j] != i:
                rep = representatives[j]
                if vacancies[rep] > 0:
                    indices_out[i, j] = indices[i, j]
                    vacancies[rep] -= 1
                    if not np.any(vacancies):
                        break
    return indices_out

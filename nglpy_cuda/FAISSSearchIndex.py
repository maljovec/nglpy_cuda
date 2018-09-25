"""
    A generic Search index structure that specifies the bare minimum
    functionality needed by any one implementation of an approximate
    k nearest neighbor structure
"""
import numpy as np
import faiss
from .SearchIndex import SearchIndex
from .utils import *


class FAISSSearchIndex(SearchIndex):
    """ A neighborhood graph that represents the connectivity of a given
    data matrix.

    Attributes:
        None
    """

    def __init__(self, **kwargs):
        """ Initializes the underlying algorithm with any user-provided
            parameters
        """
        pass

    def fit(self, X):
        """ Will build any supporting data structures if necessary given
            the data stored in X
        """
        self.X = X
        d = self.X.shape[1]

        self.res = faiss.StandardGpuResources()
        ################################################################
        # Approximate search

        # self.index = faiss.index_factory(d, ",IVF16384,PQ5")
        # self.index = faiss.index_factory(d, "IVF65536,FlatL2")
        # self.index = faiss.index_factory(d, "IDMap,FlatL2")
        # Does not work due to out of memory issue
        # self.index = faiss.index_factory(d, "Flat")

        # faster, uses more memory
        # self.index = faiss.index_factory(d, "IVF131072,Flat")
        # self.index = faiss.index_factory(d, "IVF16384,Flat")

        # co = faiss.GpuClonerOptions()

        # here we are using a 64-byte PQ, so we must set the lookup
        # tables to 16 bit float (this is due to the limited temporary
        # memory).
        # co.useFloat16 = True
        # self.index = faiss.index_cpu_to_gpu(res, 0, index, co)
        ################################################################
        # Exact search
        self.flat_config = faiss.GpuIndexFlatConfig()
        self.flat_config.device = 0
        self.flat_config.useFloat16 = True

        self.index = faiss.GpuIndexFlatL2(self.res, d, self.flat_config)
        ################################################################

        self.index.nprobe = 1  # 256

        self.index.train(X)
        self.index.add(X)

    def search(self, idx, k, return_distance=True):
        """ Returns the list of neighbors associated to one or more
            poiints in the dataset.

        Args:
            idx: one or more indices in X for which we want to retrieve
                 neighbors.

        Returns:
            A numpy array of the k nearest neighbors to each input point

            A numpy array specifying the distances to each neighbor
        """
        test_X = np.atleast_2d(self.X[idx, :])
        distance_matrix, edge_matrix = self.index.search(test_X, k)
        edge_matrix = np.array(edge_matrix, dtype=i32)
        if return_distance:
            distance_matrix = np.array(distance_matrix, dtype=f32)
            return distance_matrix, edge_matrix
        else:
            return edge_matrix

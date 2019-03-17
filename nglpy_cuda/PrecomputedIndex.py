"""
    A generic Search index structure that specifies the bare minimum
    functionality needed by any one implementation of an approximate
    k nearest neighbor structure
"""
import numpy as np
from .SearchIndex import SearchIndex
from .utils import i32


class SKLSearchIndex(SearchIndex):
    """ A neighborhood graph that represents the connectivity of a given
    data matrix.

    Attributes:
        None
    """

    def __init__(self, k_file):
        """ Initializes the underlying algorithm with any user-provided
            parameters
        """
        self.edges = np.loadtxt(k_file, dtype=int)

    def fit(self, X):
        """ Will build any supporting data structures if necessary given
            the data stored in X
        """
        self.X = X

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
        edge_matrix = np.array(self.edges[idx], dtype=i32)
        if return_distance:
            # TODO
            distance_matrix = np.zeros(edge_matrix.shape)
            return distance_matrix, edge_matrix
        else:
            return edge_matrix

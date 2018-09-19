"""
    The API for using NGLPy with CUDA
"""
from threading import Thread
from queue import Queue, Empty

import nglpy_cuda as ngl
import numpy as np

from .utils import *
from .SKLSearchIndex import SKLSearchIndex


class Graph(object):
    """ A neighborhood graph that represents the connectivity of a given
    data matrix.

    Attributes:
        None
    """

    def __init__(self,
                 X,
                 index=None,
                 max_neighbors=-1,
                 relaxed=False,
                 beta=1,
                 p=2.0,
                 discrete_steps=-1):
        """Initialization of the graph object. This will convert all of
        the passed in parameters into parameters the C++ implementation
        of NGL can understand and then issue an external call to that
        library.

        Args:
            X (matrix): The data matrix for which we will be determining
                connectivity.
            index (string): A nearest neighbor index structure which can
                be queried and pruned
            max_neighbors (int): The maximum number of neighbors to
                associate with any single point in the dataset.
            relaxed (bool): Whether the relaxed ERG should be computed
            beta (float): Defines the shape of the beta skeleton
            p (float): The Lp-norm to use in computing the shape
            discrete_steps (int): The number of steps to use if using
                the discrete version. -1 (default) signifies to use the
                continuous algorithm.
        """
        self.X = np.array(X, dtype=f32)
        N = len(self.X)

        if max_neighbors < 0:
            self.max_neighbors = min(1000, N)
        else:
            self.max_neighbors = max_neighbors

        self.relaxed = relaxed
        self.beta = beta
        self.p = p
        self.discrete_steps = discrete_steps

        if index is None:
            self.nn_index = SKLSearchIndex()
        else:
            self.nn_index = index
        self.nn_index.fit(self.X)

        self.query_size = int(min(1e6 // self.max_neighbors, N))

        self.edge_list = Queue(self.query_size*10)
        self.done = False

        Thread(target=self.populate).start()

    def populate(self):
        start_index = 0
        chunked = self.X.nbytes > 1e9
        while start_index < self.X.shape[0]:
            end_index = start_index+self.query_size
            working_set = np.array(range(start_index, end_index))

            distances, edge_matrix = self.nn_index.search(working_set,
                                                          self.max_neighbors)
            if chunked:

                # It is possible that we cannot store the entirety of X on
                # the GPU, so figure out the subset of Xs and carefully
                # replace the edge_matrix values
                indices = np.unique(
                    np.hstack((working_set, edge_matrix.flatten())))
                X = self.X[indices, :]

                # Create a lookup for the new indices in the sub-array
                index_map = {}
                for i, key in enumerate(indices):
                    index_map[key] = i

                for i in range(edge_matrix.shape[0]):
                    for j in range(edge_matrix.shape[1]):
                        if edge_matrix[i, j] != -1:
                            edge_matrix[i, j] = index_map[edge_matrix[i, j]]
            else:
                X = self.X

            if self.discrete_steps > 0:
                edge_matrix = ngl.prune_discrete(X, edge_matrix,
                                                 relaxed=self.relaxed,
                                                 steps=self.discrete_steps,
                                                 beta=self.beta, lp=self.p)
            else:
                edge_matrix = ngl.prune(X, edge_matrix, self.relaxed,
                                        self.beta, self.p)

            if chunked:
                # Reverse the lookup to the original indices of the whole array
                index_map = {}
                for key, i in enumerate(indices):
                    index_map[key] = i

                for i in range(edge_matrix.shape[0]):
                    for j in range(edge_matrix.shape[1]):
                        if edge_matrix[i, j] != -1:
                            edge_matrix[i, j] = index_map[edge_matrix[i, j]]

            for i, row in enumerate(edge_matrix):
                p_index = start_index+i
                for j, q_index in enumerate(row):
                    if q_index != -1:
                        self.edge_list.put(
                            (p_index, q_index, distances[i, j]))
            start_index += self.query_size
        self.done = True

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        while not self.edge_list.empty() or not self.done:
            try:
                next_edge = self.edge_list.get(timeout=1)
                return next_edge
            except Empty:
                pass
        raise StopIteration

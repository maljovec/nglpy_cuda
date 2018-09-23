"""
    The API for using NGLPy with CUDA
"""
from threading import Thread
from queue import Queue, Empty

import nglpy_cuda as ngl
import numpy as np
import time

from .utils import f32, i32
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
                 discrete_steps=-1,
                 query_size=None):
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
            query_size (int): The number of points to process with each
                call to the GPU, this should be computed based on
                available resources
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

        if query_size is None:
            # Because we are using f32 and i32:
            bytes_per_number = 4
            N, D = X.shape
            k = self.max_neighbors

            # Worst-case upper bound limit of the number of points
            # needed in a single query
            if self.relaxed:
                # If we are requesting a relaxed graph, then we don't
                # need neighbors' neighbors
                worst_case = D + 2*k
            else:
                worst_case = (D + 2*k) * (k + 1)

            # If we are using the discrete algorithm, remember we need
            # to add the template's storage as well to the GPU
            if discrete_steps > 0:
                worst_case += discrete_steps

            divisor = bytes_per_number * worst_case

            query_size = min(ngl.get_available_device_memory() // divisor, N)

        self.query_size = int(query_size)

        print('Problem Size: {}'.format(N))
        print('  Query Size: {}'.format(self.query_size))
        print('  GPU Memory: {}'.format(ngl.get_available_device_memory()))
        self.chunked = self.X.shape[0] > self.query_size
        print('     Chunked: {}'.format(self.chunked))

        self.edge_list = Queue(self.query_size*10)
        self.done = False

        Thread(target=self.populate, daemon=True).start()

    def populate_chunk(self, start_index):
        end_index = min(start_index+self.query_size, self.X.shape[0])
        count = end_index - start_index
        working_set = np.array(range(start_index, end_index))
        # print('Working Set: {}'.format(str(working_set)))
        start = time.process_time()
        distances, edge_matrix = self.nn_index.search(working_set,
                                                      self.max_neighbors)
        end = time.process_time()
        print('\tskl query: {} '.format(end-start))

        indices = working_set
        # We will need the locations of these additional points since
        # we need to know if they fall into the empty region of any
        # edges above
        additional_indices = np.setdiff1d(edge_matrix.ravel(),
                                          working_set)

        # print('Additional indices needed: {}'.format(str(additional_indices)))

        if additional_indices.shape[0] > 0:
            # It is possible that we cannot store the entirety of X on
            # the GPU, so figure out the subset of Xs and carefully
            # replace the edge_matrix values
            indices = np.hstack((working_set, additional_indices))
            if not self.relaxed:
                start = time.process_time()
                neighbor_matrix = self.nn_index.search(additional_indices,
                                                       self.max_neighbors,
                                                       False)
                end = time.process_time()
                print('\tsecondary skl query: {} s'.format(end-start))

                # We don't care about whether any of the edges of these
                # extra rows are valid yet, but the algorithm will need
                # them to prune correctly
                edge_matrix = np.vstack((edge_matrix, neighbor_matrix))

                # Since we will be using the edges above for queries, we
                # need to make sure we have the locations of everything
                # they touch
                neighbor_indices = np.setdiff1d(neighbor_matrix.ravel(),
                                                indices)
                # print('Additional neighboring indices needed: {}'.format(str(neighbor_indices)))
                if neighbor_indices.shape[0] > 0:
                    indices = np.hstack((indices, neighbor_indices))

        X = self.X[indices, :]
        start = time.process_time()
        if self.discrete_steps > 0:
            edge_matrix = ngl.prune_discrete(X,
                                             edge_matrix,
                                             indices=indices,
                                             relaxed=self.relaxed,
                                             steps=self.discrete_steps,
                                             beta=self.beta,
                                             lp=self.p,
                                             count=count)
        else:
            edge_matrix = ngl.prune(X,
                                    edge_matrix,
                                    indices=indices,
                                    relaxed=self.relaxed,
                                    beta=self.beta,
                                    lp=self.p,
                                    count=count)
        end = time.process_time()
        print('GPU time: {} s'.format(end-start))

        start = time.process_time()
        valid_edges = ngl.get_edge_list(edge_matrix, distances)
        for edge in valid_edges:
            self.edge_list.put(edge)
        end = time.process_time()
        print('\tList time: {} s'.format(end-start))

    def populate_whole(self):
        count = self.X.shape[0]
        working_set = np.array(range(count))
        distances, edge_matrix = self.nn_index.search(working_set,
                                                      self.max_neighbors)
        if self.discrete_steps > 0:
            edge_matrix = ngl.prune_discrete(self.X, edge_matrix,
                                             relaxed=self.relaxed,
                                             steps=self.discrete_steps,
                                             beta=self.beta, lp=self.p)
        else:
            edge_matrix = ngl.prune(self.X, edge_matrix,
                                    relaxed=self.relaxed,
                                    beta=self.beta,
                                    lp=self.p)

        for i, row in enumerate(edge_matrix):
            p_index = i
            for j, q_index in enumerate(row):
                if q_index != -1:
                    self.edge_list.put(
                        (p_index, q_index, distances[i, j]))

    def populate(self):
        try:
            if self.chunked:
                start_index = 0
                while start_index < self.X.shape[0]:
                    start = time.process_time()
                    self.populate_chunk(start_index)
                    end = time.process_time()
                    print('populate chunk {} s'.format(end-start))
                    start_index += self.query_size
            else:
                self.populate_whole()
            self.done = True
        except Exception as e:
            # Signal the main thread that we won't be sending any more
            # data before raising the exception
            self.done = True
            raise e

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

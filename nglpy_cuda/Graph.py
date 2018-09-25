"""
    The API for using NGLPy with CUDA
"""
from threading import Thread
from queue import Queue, Empty

import nglpy_cuda as ngl
import numpy as np

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

        available_gpu_memory = ngl.get_available_device_memory()
        if query_size is None:
            # Because we are using f32 and i32:
            bytes_per_number = 4
            N, D = X.shape
            k = self.max_neighbors

            # Worst-case upper bound limit of the number of points
            # needed in a single query
            if self.relaxed:
                # For the relaxed algorithm we need n*D storage for the
                # point locations plus n*k for the representation of the
                # input edges plus another n*k for the output edges
                # We could potentially only need one set of edges for
                # this version
                worst_case = D + 2*k
            else:
                # For the strict algorithm, if we are processing n
                # points at a time, we need the point locations of all
                # of their neigbhors, thus in the worst case, we need
                # n + n*k point locations and rows of the edge matrix.
                # the 2*k again represents the need for two versions of
                # the edge matrix. Here, we definitely need an input and
                # an output array
                worst_case = (D + 2*k) * (k + 1)

            # If we are using the discrete algorithm, remember we need
            # to add the template's storage as well to the GPU
            if discrete_steps > 0:
                available_gpu_memory -= discrete_steps*bytes_per_number

            divisor = bytes_per_number * worst_case

            query_size = min(available_gpu_memory // divisor, N)

        self.query_size = int(query_size)

        self.chunked = self.X.shape[0] > self.query_size
        # print('Problem Size: {}'.format(N))
        # print('  Query Size: {}'.format(self.query_size))
        # print('  GPU Memory: {}'.format(available_gpu_memory))
        # print('     Chunked: {}'.format(self.chunked))

        self.edge_list = Queue(self.query_size*10)
        self.done = False

        Thread(target=self.populate, daemon=True).start()

    def populate_chunk(self, start_index):
        end_index = min(start_index+self.query_size, self.X.shape[0])
        count = end_index - start_index
        working_set = np.array(range(start_index, end_index))

        distances, edges = self.nn_index.search(working_set,
                                                self.max_neighbors)

        indices = working_set
        # We will need the locations of these additional points since
        # we need to know if they fall into the empty region of any
        # edges above
        additional_indices = np.setdiff1d(edges.ravel(),
                                          working_set)

        if additional_indices.shape[0] > 0:
            # It is possible that we cannot store the entirety of X
            # and edges on the GPU, so figure out the subset of Xs and
            # carefully replace the edges values
            indices = np.hstack((working_set, additional_indices))
            if not self.relaxed:
                neighbor_edges = self.nn_index.search(additional_indices,
                                                      self.max_neighbors,
                                                      False)

                # We don't care about whether any of the edges of these
                # extra rows are valid yet, but the algorithm will need
                # them to prune correctly
                edges = np.vstack((edges, neighbor_edges))

                # Since we will be using the edges above for queries, we
                # need to make sure we have the locations of everything
                # they touch
                neighbor_indices = np.setdiff1d(neighbor_edges.ravel(),
                                                indices)

                if neighbor_indices.shape[0] > 0:
                    indices = np.hstack((indices, neighbor_indices))

        indices = indices.astype(i32)
        X = self.X[indices, :]
        edges = ngl.prune(X,
                          edges,
                          indices=indices,
                          relaxed=self.relaxed,
                          steps=self.discrete_steps,
                          beta=self.beta,
                          lp=self.p,
                          count=count)

        valid_edges = ngl.get_edge_list(edges[:count], distances[:count], indices)
        for edge in valid_edges:
            self.edge_list.put(edge)

    def populate_whole(self):
        count = self.X.shape[0]
        working_set = np.array(range(count))
        distances, edges = self.nn_index.search(working_set,
                                                self.max_neighbors)

        edges = ngl.prune(self.X,
                          edges,
                          relaxed=self.relaxed,
                          steps=self.discrete_steps,
                          beta=self.beta,
                          lp=self.p)

        valid_edges = ngl.get_edge_list(edges, distances)
        for edge in valid_edges:
            self.edge_list.put(edge)

    def populate(self):
        try:
            if self.chunked:
                start_index = 0
                while start_index < self.X.shape[0]:
                    self.populate_chunk(start_index)
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

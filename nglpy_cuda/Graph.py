"""
    The API for using NGLPy with CUDA
"""
import nglpy_cuda as ngl
import numpy as np

from .utils import f32, i32
from .SKLSearchIndex import SKLSearchIndex

from threading import Thread
import sys
import os
import abc
if sys.version_info.major >= 3:
    from queue import Queue, Empty
    ABC = abc.ABC
else:
    from Queue import Queue, Empty
    ABC = abc.ABCMeta('ABC', (), {})


class Graph(ABC):
    """ A neighborhood graph that represents the connectivity of a given
    data matrix.

    Attributes:
        None
    """

    def __init__(self,
                 index=None,
                 max_neighbors=-1,
                 query_size=None,
                 cached=False):
        """Initialization of the graph object. This will convert all of
        the passed in parameters into parameters the C++ implementation
        of NGL can understand and then issue an external call to that
        library.

        Args:
            index (string): A nearest neighbor index structure which can
                be queried and pruned
            max_neighbors (int): The maximum number of neighbors to
                associate with any single point in the dataset.
            query_size (int): The number of points to process with each
                call to the GPU, this should be computed based on
                available resources
            cached (bool): Flag denoting whether the resulting computation
                should be stored for future use, will consume more memory,
                but can drastically reduce the runtime for subsequent
                iterations through the data.
        """
        self.X = np.array([[]], dtype=f32)
        self.max_neighbors = max_neighbors
        if index is None:
            self.nn_index = SKLSearchIndex()
        else:
            self.nn_index = index
        self.query_size = query_size
        self.cached = cached
        self.edges = None

    def build(self, X):
        """ Build a graph based on the incoming data

        Args:
            X (matrix): The data matrix for which we will be determining
                connectivity.
        """
        self.X = np.array(X, dtype=f32)
        N = len(X)

        if self.max_neighbors < 0:
            self.max_neighbors = min(1000, N)

        self.nn_index.fit(self.X)

        if self.query_size is None:
            self.query_size = self.compute_query_size()

        self.chunked = self.X.shape[0] > self.query_size

        # print('Problem Size: {}'.format(N))
        # print('  Query Size: {}'.format(self.query_size))
        # print('  GPU Memory: {}'.format(available_gpu_memory))
        # print('     Chunked: {}'.format(self.chunked))
        sys.stdout.flush()

        self.edge_list = Queue(self.query_size*10)
        self.needs_reset = False

        self.worker_thread = Thread(target=self.populate)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    @abc.abstractmethod
    def compute_query_size(self):
        pass

    @abc.abstractmethod
    def collect_additional_indices(self, edges, indices):
        pass

    @abc.abstractmethod
    def prune(self, X, edges, indices=None, count=None):
        pass

    def populate_chunk(self, start_index):
        end_index = min(start_index+self.query_size, self.X.shape[0])
        count = end_index - start_index
        working_set = np.array(range(start_index, end_index))

        edges = self.nn_index.search(working_set, self.max_neighbors, False)

        indices = self.collect_additional_indices(edges, working_set)
        indices = np.array(indices, dtype=i32)
        X = self.X[indices, :]

        edges = self.prune(X, edges, indices, count)

        # We will cache these for later use
        if self.cached:
            self.edges[start_index:end_index, :] = edges[:count]

        # Since, we are taking a lot of time to generate these, then we
        # should give the user something to process in the meantime, so
        # don't remove these lines and make sure to return in the main
        # populate before the same process is done again.
        self.push_edges(edges[:count], indices[:count])

    def populate_whole(self):
        count = self.X.shape[0]
        working_set = np.array(range(count))
        edges = self.nn_index.search(working_set, self.max_neighbors, False)

        edges = self.prune(self.X, edges)

        if self.cached:
            self.edges = edges
        self.push_edges(edges)

    def populate(self):
        point_count = self.X.shape[0]
        start_index = 0
        if self.edges is not None:
            start_index = point_count

        fname = 'nglpy.checkpoint'
        if self.cached and os.path.isfile(fname):
            with open(fname, 'r') as f:
                start_index = int(f.read())

        if self.edges is None or start_index < point_count:
            data_shape = (point_count, self.max_neighbors)
            if self.cached:
                self.edges = np.memmap(
                    'edges.npy', dtype=i32, mode='w+', shape=data_shape)

            if self.chunked:
                while start_index < point_count:
                    self.populate_chunk(start_index)
                    start_index += self.query_size
                    start_index = min(start_index, point_count)
                    with open(fname, 'w') as f:
                        f.write(str(start_index))
                    print('Checkpoint: {}'.format(start_index))
            else:
                self.populate_whole()
        else:
            self.push_edges(self.edges)

    def push_edges(self, edges, indices=None):
        if indices is not None:
            valid_edges = ngl.get_edge_list(edges, indices)
        else:
            valid_edges = ngl.get_edge_list(edges)
        for edge in valid_edges:
            self.edge_list.put(edge)

    def restart_iteration(self):
        if not self.worker_thread.is_alive() and self.needs_reset:
            self.edge_list.queue.clear()
            self.needs_reset = False
            self.worker_thread = Thread(target=self.populate)
            self.worker_thread.daemon = True
            self.worker_thread.start()

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.needs_reset:
            self.restart_iteration()
        while not self.edge_list.empty() or self.worker_thread.is_alive():
            try:
                next_edge = self.edge_list.get(timeout=1)
                return next_edge
            except Empty:
                pass
        self.needs_reset = True
        raise StopIteration

    def full_graph(self):
        neighborhoods = {}
        for (p, q, d) in self:
            p = int(p)
            q = int(q)
            if p not in neighborhoods:
                neighborhoods[p] = set()
            if q not in neighborhoods:
                neighborhoods[q] = set()
            neighborhoods[p].add(q)
            neighborhoods[q].add(p)
        return neighborhoods

    def neighbors(self, i):
        nn = []
        if self.edges is None:
            raise NotImplementedError

        for value in self.edges[i]:
            if value != -1:
                nn.append(value)
        return nn

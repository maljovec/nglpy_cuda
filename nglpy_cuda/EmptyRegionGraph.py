"""
    The API for using NGLPy with CUDA
"""
import nglpy_cuda as ngl
import numpy as np

from .utils import f32, i32
from .SKLSearchIndex import SKLSearchIndex
from .Graph import Graph

import time
import psutil
from threading import Thread
import sys
import os
if sys.version_info.major >= 3:
    from queue import Queue, Empty
else:
    from Queue import Queue, Empty

class EmptyRegionGraph(Graph):
    """ A neighborhood graph that represents the connectivity of a given
    data matrix.

    Attributes:
        None
    """

    def __init__(self,
                 index=None,
                 max_neighbors=-1,
                 relaxed=False,
                 beta=1,
                 p=2.0,
                 discrete_steps=-1,
                 query_size=None,
                 cached=True):
        """Initialization of the graph object. This will convert all of
        the passed in parameters into parameters the C++ implementation
        of NGL can understand and then issue an external call to that
        library.

        Args:
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
            cached (bool): Flag denoting whether the resulting computation
                should be stored for future use, will consume more memory,
                but can drastically reduce the runtime for subsequent
                iterations through the data.
        """
        super(EmptyRegionGraph, self).__init__(
            index=index,
            max_neighbors=max_neighbors,
            query_size=query_size,
            cached=cached
        )

        self.relaxed = relaxed
        self.beta = beta
        self.p = p
        self.discrete_steps = discrete_steps

    def compute_query_size(self):
        """
        """
        if self.query_size is not None:
            return self.query_size

        available_gpu_memory = ngl.get_available_device_memory()

        # Because we are using f32 and i32:
        bytes_per_number = 4
        N, D = self.X.shape
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
        if self.discrete_steps > 0:
            available_gpu_memory -= self.discrete_steps*bytes_per_number

        divisor = bytes_per_number * worst_case
        self.query_size = min(available_gpu_memory // divisor, N)
        self.query_size = int(self.query_size)
        return self.query_size

    def collect_additional_indices(self, edges, indices):
        start = time.time()
        # We will need the locations of these additional points since
        # we need to know if they fall into the empty region of any
        # edges above
        additional_indices = np.setdiff1d(edges.ravel(),
                                          indices)

        end = time.time()
        print('Calculating additional indices: {} s'.format(end-start))
        sys.stdout.flush()
        start = time.time()

        if additional_indices.shape[0] > 0:
            # It is possible that we cannot store the entirety of X
            # and edges on the GPU, so figure out the subset of Xs and
            # carefully replace the edges values
            indices = np.hstack((indices, additional_indices))

            end = time.time()
            print('Stacking indices: {} s'.format(end-start))
            sys.stdout.flush()
            start = time.time()

            if not self.relaxed:
                neighbor_edges = self.nn_index.search(additional_indices,
                                                      self.max_neighbors,
                                                      False)
                end = time.time()
                print('Secondary knn query: {} s'.format(end-start))
                sys.stdout.flush()
                start = time.time()
                # We don't care about whether any of the edges of these
                # extra rows are valid yet, but the algorithm will need
                # them to prune correctly
                edges = np.vstack((edges, neighbor_edges))

                end = time.time()
                print('Stacking edges: {} s'.format(end-start))
                sys.stdout.flush()
                start = time.time()

                # Since we will be using the edges above for queries, we
                # need to make sure we have the locations of everything
                # they touch
                neighbor_indices = np.setdiff1d(neighbor_edges.ravel(),
                                                indices)

                end = time.time()
                print('Calculating neighbor indices: {} s'.format(end-start))
                sys.stdout.flush()
                start = time.time()

                if neighbor_indices.shape[0] > 0:
                    indices = np.hstack((indices, neighbor_indices))

                end = time.time()
                print('Stacking neighboring indices: {} s'.format(end-start))
                sys.stdout.flush()
                start = time.time()

        indices = indices.astype(i32)
        return indices

    def prune(self, X, edges, indices=None):
        if indices is None:
            edges = ngl.prune(X,
                            edges,
                            relaxed=self.relaxed,
                            steps=self.discrete_steps,
                            beta=self.beta,
                            lp=self.p)
        else:
            edges = ngl.prune(X,
                    edges,
                    indices=indices,
                    relaxed=self.relaxed,
                    steps=self.discrete_steps,
                    beta=self.beta,
                    lp=self.p,
                    count=count)
        return edges
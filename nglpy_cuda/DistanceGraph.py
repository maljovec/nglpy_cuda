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

class DistanceGraph(Graph):
    """ A neighborhood graph that represents the connectivity of a given
    data matrix.

    Attributes:
        None
    """

    def __init__(self,
                 index=None,
                 max_neighbors=-1,
                 epsilon=1e-2,
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
            epsilon (float): The maximum distance of neighbors to
                associate with any single point in the dataset.
            query_size (int): The number of points to process with each
                call to the GPU, this should be computed based on
                available resources
            cached (bool): Flag denoting whether the resulting computation
                should be stored for future use, will consume more memory,
                but can drastically reduce the runtime for subsequent
                iterations through the data.
        """
        super(DistanceGraph, self).__init__(
            index=index,
            max_neighbors=max_neighbors,
            query_size=query_size,
            cached=cached
        )

        self.epsilon = epsilon

    def compute_query_size(self):
        """
        """
        if self.query_size is None:
            self.query_size = 1000000
        return self.query_size

    def collect_additional_indices(self, edges, indices):
        return indices

    def prune(self, X, edges, indices=None):
        return edges

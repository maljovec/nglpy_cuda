"""
    This module is meant to mirror the API from nglpy in order to create
    a drop-in replacement. Consider this class for deprecation due to
    inefficient handling of neighborhood queries.
"""
from threading import Thread
from queue import Queue, Empty

import nglpy_cuda as ngl
import numpy as np

from .utils import f32, i32
from .SKLSearchIndex import SKLSearchIndex
from .Graph import Graph


class ProbabilisticGraph(Graph):
    """ A probabilistic neighborhood graph that represents an uncertain
    connectivity of a given data matrix.

    Attributes:
        None
    """

    def __init__(
        self,
        X,
        steepness=3,
        index=None,
        max_neighbors=-1,
        relaxed=False,
        beta=1,
        p=2.0,
        discrete_steps=-1,
        query_size=None,
    ):
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
        self.steepness = steepness
        self.seed = 0
        super(ProbabilisticGraph, self).__init__(
            X,
            index=index,
            max_neighbors=max_neighbors,
            relaxed=relaxed,
            beta=beta,
            p=p,
            discrete_steps=discrete_steps,
            query_size=query_size,
        )

    def reseed(self, seed):
        self.seed = seed

    def populate_chunk(self, start_index):
        end_index = min(start_index + self.query_size, self.X.shape[0])
        count = end_index - start_index
        working_set = np.array(range(start_index, end_index))

        distances, edges = self.nn_index.search(working_set, self.max_neighbors)

        indices = working_set
        # We will need the locations of these additional points since
        # we need to know if they fall into the empty region of any
        # edges above
        additional_indices = np.setdiff1d(edges.ravel(), working_set)

        if additional_indices.shape[0] > 0:
            # It is possible that we cannot store the entirety of X
            # and edges on the GPU, so figure out the subset of Xs and
            # carefully replace the edges values
            indices = np.hstack((working_set, additional_indices))
            if not self.relaxed:
                neighbor_edges = self.nn_index.search(
                    additional_indices, self.max_neighbors, False
                )

                # We don't care about whether any of the edges of these
                # extra rows are valid yet, but the algorithm will need
                # them to prune correctly
                edges = np.vstack((edges, neighbor_edges))

                # Since we will be using the edges above for queries, we
                # need to make sure we have the locations of everything
                # they touch
                neighbor_indices = np.setdiff1d(neighbor_edges.ravel(), indices)

                if neighbor_indices.shape[0] > 0:
                    indices = np.hstack((indices, neighbor_indices))

        indices = indices.astype(i32)
        X = self.X[indices, :]
        probabilities = ngl.associate_probability(
            X,
            edges,
            steepness=self.steepness,
            indices=indices,
            relaxed=self.relaxed,
            steps=self.discrete_steps,
            beta=self.beta,
            lp=self.p,
            count=count,
        )

        mask = np.random.binomial(1, 1-probabilities[:count]).astype(bool)
        edges[mask] = -1
        valid_edges = ngl.get_edge_list(
            edges[:count], distances[:count], indices, mask
        )
        for edge in valid_edges:
            self.edge_list.put(edge)

    def populate_whole(self):
        count = self.X.shape[0]
        working_set = np.array(range(count))
        distances, edges = self.nn_index.search(working_set, self.max_neighbors)

        probabilities = ngl.associate_probability(
            self.X,
            edges,
            steepness=self.steepness,
            relaxed=self.relaxed,
            steps=self.discrete_steps,
            beta=self.beta,
            lp=self.p,
        )
        mask = np.random.binomial(1, 1 - probabilities).astype(bool)
        edges[mask] = -1
        valid_edges = ngl.get_edge_list(edges, distances)
        for edge in valid_edges:
            self.edge_list.put(edge)

    def populate(self):
        np.random.seed(self.seed)
        super(ProbabilisticGraph, self).populate()

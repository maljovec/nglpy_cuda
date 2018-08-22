"""
    This module is meant to mirror the API from nglpy in order to create
    a drop-in replacement. Consider this class for deprecation due to
    inefficient handling of neighborhood queries.
"""
import sklearn.neighbors
import nglpy_cuda
import numpy as np

f32 = np.float32
i32 = np.int32


class Graph(object):
    """ A neighborhood graph that represents the connectivity of a given
    data matrix.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be
    documented inline with the attribute's declaration (see __init__
    method below).

    Properties created with the ``@property`` decorator should be
    documented in the property's getter method.

    Attributes:
        None
    """

    def __init__(self, X, graph, max_neighbors, beta, edges=None,
                 connect=False, p=2.0, discrete_steps=-1):
        """Initialization of the graph object. This will convert all of
        the passed in parameters into parameters the C++ implementation
        of NGL can understand and then issue an external call to that
        library.

        Args:
            X (matrix): The data matrix for which we will be determining
                connectivity.
            graph (string): The type of graph to construct.
            max_neighbors (int): The maximum number of neighbors to
                associate with any single point in the dataset.
            beta (float): Only relevant when the graph type is a "beta
                skeleton"
            edges (list): A list of pre-defined edges to prune
            connect (boolean): A flag specifying whether the data should
                be a single connected component, this feature is not yet
                implemented in the GPU version, so it must be false.
            p (float): The Lp-norm to use in computing the shape
            discrete_steps (int): The number of steps to use if using
                the discrete version. -1 (default) signifies to use the
                continuous algorithm.
        """
        self.X = np.array(X, dtype=f32)
        N = len(self.X)
        D = len(self.X[0])

        if connect:
            raise NotImplementedError("The connect feature is not yet "
                                      "implemented in the GPU version "
                                      "of nglpy.")

        if edges is None:
            # They want us to build the starting graph
            if max_neighbors <= 0:
                max_neighbors = N - 1
            knnAlgorithm = sklearn.neighbors.NearestNeighbors(max_neighbors+1)
            knnAlgorithm.fit(X)
            edge_matrix = np.array(knnAlgorithm.kneighbors(
                X, return_distance=False), dtype=i32)
        elif len(edges) != len(X) or not hasattr(edges[0], 'len'):
            # If the length of edges is not the same as X, then this
            # cannot be an edge matrix, it must be a list of edges.
            # There is a weird edge case here where they could have
            # an edge list that is exactly the same length as the number
            # of points. We are going to assume that if the size of each
            # edge is two it is probably an edge list, who would want to
            # prune a 2-nearest neighbor graph with this method?

            # Assume nothing the user gave you was correct
            edge_counts = np.zeros(len(X), dtype=i32)
            edge_set = set()
            for i in range(0, len(edges), 2):
                e = (edges[i], edges[i+1])
                lo = min(e)
                hi = max(e)
                if lo != hi and (lo, hi) not in edge_set:
                    edge_counts[lo] += 1
                    edge_counts[hi] += 1
                    edge_set.add((lo, hi))

            max_neighbors = np.max(edge_counts)
            # Initialize everything as completely disjoint
            edge_matrix = np.zeros(shape=(N, max_neighbors), dtype=i32) - 1
            edge_counts = np.zeros(len(X), dtype=i32)
            for e in edge_set:
                row1 = e[0]
                col1 = edge_counts[e[0]]
                edge_matrix[row1, col1] = e[1]
                edge_counts[e[0]] += 1

                row2 = e[1]
                col2 = edge_counts[e[1]]
                edge_matrix[row2, col2] = e[0]
                edge_counts[e[1]] += 1
        else:
            # They gave us exactly what we need to start with
            edge_matrix = np.array(edges, dtype=i32)

        k = len(edge_matrix[0])

        self.edge_matrix = nglpy_cuda.prune(
            N, D, k, p, beta, self.X, edge_matrix)

    def neighbors(self, idx=None):
        """ Returns the list of neighbors associated to a particular
            index in the dataset, if one is provided, otherwise a full
            dictionary is provided relating each index to a set of
            connected indices.

        Args:
            idx: (optional) a single index of the point in the input
                data matrix for which we want to retrieve neighbors.

        Returns:
            A list of indices connected to either the provided input
            index, or a dictionary where the keys are the indices in the
            whole dataset and the values are sets of indices connected
            to the key index.
        """
        if idx is None:
            edge_dict = {}
            for i in range(len(self.X)):
                edge_dict[i] = []
                for j in self.edge_matrix[i]:
                    if j != -1:
                        edge_dict[i].append(j)
                for j, _ in enumerate(self.edge_matrix):
                    if j == i or j in edge_dict[i]:
                        continue
                    for k in self.edge_matrix[j]:
                        if k == i:
                            edge_dict[i].append(j)
            for k, v in edge_dict.items():
                edge_dict[k] = tuple(v)
            return edge_dict
        else:
            ret_list = []
            for j in self.edge_matrix[idx]:
                if j != -1:
                    ret_list.append(j)

            for j, _ in enumerate(self.edge_matrix):
                if j == idx or j in ret_list:
                    continue
                for i in self.edge_matrix[j]:
                    if i == idx:
                        ret_list.append(j)

            return ret_list

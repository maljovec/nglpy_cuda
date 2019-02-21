"""
    The API for using NGLPy with CUDA
"""
import nglpy_cuda as ngl
from .Graph import Graph
from .utils import f32, i32
import samply
import numpy as np


class ConeGraph(Graph):
    """ A neighborhood graph that represents the connectivity of a given
    data matrix.

    Attributes:
        None
    """
    available_algorithms = ["yao", "theta"]

    def __init__(self,
                 index=None,
                 max_neighbors=-1,
                 num_sectors=6,
                 points_per_sector=1,
                 algorithm="yao",
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
            num_sectors (int): The number of roughly equal sized conic
                sectors that the space around the point should be split
            points_per_sector (int): The maximum number of nearest
                neighbors to keep per sector
            algorithm (str): a string representation of the type of
                graph to compute should be yao or theta
            query_size (int): The number of points to process with each
                call to the GPU, this should be computed based on
                available resources
            cached (bool): Flag denoting whether the resulting computation
                should be stored for future use, will consume more memory,
                but can drastically reduce the runtime for subsequent
                iterations through the data.
        """
        super(ConeGraph, self).__init__(
            index=index,
            max_neighbors=max_neighbors,
            query_size=query_size,
            cached=cached
        )

        self.num_sectors = num_sectors
        self.points_per_sector = points_per_sector
        self.algorithm = algorithm.strip().lower()
        if self.algorithm not in ConeGraph.available_algorithms:
            raise ValueError("Unknown algorithm specified: {}. Must be one of {}".format(
                self.algorithm, ConeGraph.available_algorithms))

    def compute_query_size(self):
        """
        """
        if self.query_size is not None:
            return self.query_size
        return 1000000

    def collect_additional_indices(self, edges, indices):
        indices = indices.astype(i32)
        return indices

    def prune(self, X, edges, indices=None, count=None):
        if self.algorithm == "yao":
            # Move this code up once the theta graph is on the gpu
            D = X.shape[1]
            vectors = samply.directional.cvt(self.num_sectors, D)
            vectors = np.array(vectors, dtype=f32)
            #####
            if indices is None:
                edges = ngl.prune_yao(X, vectors, edges, points_per_sector=self.points_per_sector)
            else:
                edges = ngl.prune_yao(X, vectors, edges, indices, count, self.points_per_sector)
            return edges
        elif self.algorithm == "theta":
            return ngl.theta_graph(X, self.num_sectors, self.points_per_sector, edges)

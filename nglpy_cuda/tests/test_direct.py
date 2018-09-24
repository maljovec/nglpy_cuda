import unittest
import numpy as np
import nglpy_cuda as ngl
# import objgraph
import os

f32 = np.float32
i32 = np.int32


def create_edge_set(edge_matrix):
    """ A helper method for putting the output edge_matrices into a
        format that is easy to test with the ground truth.
    """
    edge_set = set()
    for i, row in enumerate(edge_matrix):
        for j in row:
            if j != -1:
                edge_set.add((min(i, j), max(i, j)))
    return edge_set


class TestAPI(unittest.TestCase):
    """ Class for testing the direct to CUDA API
    """

    def setup(self):
        """
        Setup repeatable test case with a known ground truth
        """
        # User-editable variables
        dir_path = os.path.dirname(os.path.realpath(__file__))
        X = np.loadtxt(os.path.join(dir_path, 'data', 'points.txt'))
        self.X = np.array(X, dtype=f32)
        edges = np.loadtxt(os.path.join(dir_path, 'data', 'edges.txt'))
        self.edges = np.array(edges, dtype=i32)

        gold = np.loadtxt(os.path.join(dir_path, 'data',
                                       'gold_edges_strict.txt'))
        self.gold_strict = set()
        for edge in gold:
            lo, hi = min(edge), max(edge)
            self.gold_strict.add((lo, hi))

        gold = np.loadtxt(os.path.join(dir_path, 'data',
                                       'gold_edges_relaxed.txt'))
        self.gold_relaxed = set()
        for edge in gold:
            lo, hi = min(edge), max(edge)
            self.gold_relaxed.add((lo, hi))

    def test_min_distance_from_edge(self):
        """
        Testing min_distance_from_edge function
        """
        self.assertEqual(ngl.min_distance_from_edge(0, 1, 2), 0.5, '')
        self.assertEqual(ngl.min_distance_from_edge(1, 1, 2), 0., '')

    def test_create_template(self):
        """
        Testing create_template function
        """
        self.setup()
        template = ngl.create_template(1, 2, 10)
        self.assertEqual(len(template), 10, '')
        for i in range(len(template)-1):
            self.assertEqual(template[i] > template[i+1], True, '')

    def test_prune(self):
        """
        Testing prune function
        """
        self.setup()
        ngl_edges = ngl.prune(self.X, self.edges)
        edge_set = create_edge_set(ngl_edges)
        self.assertEqual(len(self.gold_strict ^ edge_set), 0, '')

    def test_prune_discrete(self):
        """
        Testing prune_discrete function with beta/lp specified.
        """
        self.setup()
        ngl_edges = ngl.prune(self.X, self.edges, steps=100)
        edge_set = create_edge_set(ngl_edges)
        self.assertEqual(len(self.gold_strict ^ edge_set), 0, '')

    def test_prune_discrete_template(self):
        """
        Testing prune_discrete fucntion with template specified
        """
        self.setup()
        template = np.array(ngl.create_template(1, 2, 100), dtype=f32)
        ngl_edges = ngl.prune(self.X, self.edges, template=template)
        edge_set = create_edge_set(ngl_edges)
        self.assertEqual(len(self.gold_strict ^ edge_set), 0, '')

    def test_get_edge_list(self):
        """
        Testing get_edge_list function
        """
        self.setup()
        ngl_edges = ngl.prune(self.X, self.edges)
        edge_list = ngl.get_edge_list(ngl_edges, np.zeros(ngl_edges.shape))
        edge_set = set()
        for (p, q, d) in edge_list:
            lo = min(p, q)
            hi = max(p, q)
            edge_set.add((lo, hi))
        self.assertEqual(len(self.gold_strict ^ edge_set), 0, '')

    # def test_memory_management(self):
    #     """ Test all functions to ensure there is no memory leakage
    #     """
    #     # I just want to make sure I am using the python reference counters
    #     # correctly by doing some memory debugging with objgraph
    #     _ = objgraph.growth(limit=None)
    #     current = objgraph.growth(limit=None)
    #     print(ngl.get_available_device_memory())
    #     print(current)
    #     self.test_min_distance_from_edge()
    #     print(ngl.get_available_device_memory())
    #     print(objgraph.growth(limit=None))
    #     # self.assertEqual(current, objgraph.growth(limit=None),
    #     #                  'There should be no memory changes.')
    #     self.test_create_template()
    #     print(ngl.get_available_device_memory())
    #     print(objgraph.growth(limit=None))
    #     # self.assertEqual(current, objgraph.growth(limit=None),
    #     #                  'There should be no memory changes.')
    #     self.test_prune()
    #     print(ngl.get_available_device_memory())
    #     print(objgraph.growth(limit=None))
    #     # self.assertEqual(current, objgraph.growth(limit=None),
    #     #                  'There should be no memory changes.')
    #     self.test_prune_discrete()
    #     print(ngl.get_available_device_memory())
    #     print(objgraph.growth(limit=None))
    #     # self.assertEqual(current, objgraph.growth(limit=None),
    #     #                  'There should be no memory changes.')
    #     self.test_get_edge_list()
    #     print(ngl.get_available_device_memory())
    #     print(objgraph.growth(limit=None))
    #     # self.assertEqual(current, objgraph.growth(limit=None),
    #     #                  'There should be no memory changes.')

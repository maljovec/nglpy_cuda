import unittest
import numpy as np
import nglpy_cuda
# import objgraph
import os

f32 = np.float32


def create_edge_set(edge_matrix):
    """ A helper method for putting the output edge_matrices into a
        format that is easy to test with the ground truth.
    """
    edge_set = set()
    for i, row in enumerate(edge_matrix):
        for j in row:
            if j != -1:
                lo, hi = (min([i, j]), max([i, j]))
                edge_set.add((lo, hi))
    return edge_set


class TestAPI(unittest.TestCase):
    """ Class for testing the direct to CUDA API
    """

    def setup(self):
        """ Setup repeatable test case with a known ground truth
        """
        # User-editable variables
        self.D = 2
        self.k = self.N = 10
        self.beta = 1
        self.p = 2
        self.steps = 50

        S = 0
        np.random.seed(S)
        # numpy random doesn't give float32 datatypes as an option, so
        # we have to hack it together a bit
        shape = (self.N, self.D)
        self.X = np.empty(shape=shape, dtype=f32)
        self.X[...] = np.random.randint(0, 1000., size=shape) / f32(1000)

        # For starters, just use a dense graph and prune it.
        self.edges = np.empty(shape=(self.N, self.N), dtype=np.int32)
        for i in range(self.N):
            self.edges[i] = np.array(range(self.N))

        self.ground_truth = set()
        GROUND_TRUTH = os.path.join(os.path.dirname(__file__),
                                    'data', 'ground_truth.txt')
        with open(GROUND_TRUTH) as f:
            for line in f:
                tokens = line.strip().split(' ')
                i = int(tokens[0])
                j = int(tokens[1])
                if j < i:
                    j, i = i, j
                self.ground_truth.add((i, j))

    def test_min_distance_from_edge(self):
        """ Testing min_distance_from_edge function

        """
        self.assertEqual(nglpy_cuda.min_distance_from_edge(0, 1, 2), 0.5, '')
        self.assertEqual(nglpy_cuda.min_distance_from_edge(1, 1, 2), 0., '')

    def test_create_template(self):
        """ Testing create_template function
        """
        self.setup()
        template = nglpy_cuda.create_template(self.beta, self.p, self.steps)
        self.assertEqual(len(template), self.steps, '')
        for i in range(len(template)-1):
            self.assertEqual(template[i] > template[i+1], True, '')

    def test_prune(self):
        """ Testing prune function
        """
        self.setup()
        ngl_edges = nglpy_cuda.prune(self.N, self.D, self.k, self.p, self.beta,
                                     self.X, self.edges)
        edge_set = create_edge_set(ngl_edges)
        self.assertEqual(len(self.ground_truth ^ edge_set), 0, '')

    def test_prune_discrete(self):
        """ Testing prune_discrete function with beta/lp specified.
        """
        self.setup()
        ngl_edges = nglpy_cuda.prune_discrete(self.N, self.D, self.k,
                                              self.steps, self.beta, self.p,
                                              self.X, self.edges)
        edge_set = create_edge_set(ngl_edges)
        self.assertEqual(len(self.ground_truth ^ edge_set), 0, '')

    def test_prune_discrete_template(self):
        """ Testing prune_discrete fucntion with template specified
        """
        self.setup()
        template = np.array(nglpy_cuda.create_template(self.beta,
                                                       self.p,
                                                       self.steps), dtype=f32)
        ngl_edges = nglpy_cuda.prune_discrete(self.N, self.D, self.k,
                                              self.steps, template, self.X,
                                              self.edges)
        edge_set = create_edge_set(ngl_edges)
        self.assertEqual(len(self.ground_truth ^ edge_set), 0, '')

    def test_get_edge_list(self):
        """ Testing get_edge_list function
        """
        self.setup()
        ngl_edges = nglpy_cuda.prune(self.N, self.D, self.k, self.p, self.beta,
                                     self.X, self.edges)
        edge_list = nglpy_cuda.get_edge_list(self.N, self.k, ngl_edges)
        edge_set = set()
        for item in edge_list:
            lo = min(item)
            hi = max(item)
            edge_set.add((lo, hi))
        self.assertEqual(len(self.ground_truth ^ edge_set), 0, '')

    # def test_memory_management(self):
    #     """ Test all functions to ensure there is no memory leakage
    #     """
    #     # I just want to make sure I am using the python reference counters
    #     # correctly by doing some memory debugging with objgraph
    #     _ = objgraph.growth(limit=None)
    #     current = objgraph.growth(limit=None)
    #     self.test_min_distance_from_edge()
    #     self.assertEqual(current, objgraph.growth(limit=None),
    #                      'There should be no memory changes.')
    #     self.test_create_template()
    #     self.assertEqual(current, objgraph.growth(limit=None),
    #                      'There should be no memory changes.')
    #     self.test_prune()
    #     self.assertEqual(current, objgraph.growth(limit=None),
    #                      'There should be no memory changes.')
    #     self.test_prune_discrete()
    #     self.assertEqual(current, objgraph.growth(limit=None),
    #                      'There should be no memory changes.')
    #     self.test_get_edge_list()
    #     self.assertEqual(current, objgraph.growth(limit=None),
    #                      'There should be no memory changes.')

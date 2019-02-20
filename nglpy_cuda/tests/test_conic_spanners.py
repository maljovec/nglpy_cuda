""" This module will test the functionality of nglpy.Graph when using the
    conic spanner graphs (Yao and Θ) type
"""
import unittest
import nglpy_cuda as ngl
import math
import samply
import numpy as np
from sklearn import neighbors


class TestConics(unittest.TestCase):
    """ Class for testing the Yao and Θ Graphs
    """

    def setup(self):
        """ Setup function will create a fixed point set and parameter
        settings for testing different aspects of the conic spanners.
        """

        # We will only test the edges of the origin, since we know what
        # they should be in both cases
        self.points = [[0., 0.]]

        # Generate the same set of vectors that the Yao and Θ graphs
        # will use in order to test the Θ graph's ability to recover
        # points exactly on the conic axes
        count = 6
        dim = 2
        np.random.seed(0)
        vectors = samply.directional.cvt(count, dim)

        # Now take those same vectors, rotate them by one quarter of the
        # bisecting angle of each conic section to ensure they still lie
        # well within the desired conic sections and reduce their
        # magnitudes by half. These set of points should be reported
        # as the edges of the Yao graph.
        theta = 2 * math.pi / (count * 8)
        rotation_matrix = np.array(
            [
                [math.cos(theta), -math.sin(theta)],
                [math.sin(theta), math.cos(theta)],
            ]
        )
        rotated_vectors = (rotation_matrix @ vectors.T).T / 2.

        self.points = np.vstack((self.points, vectors, rotated_vectors))

        nn = neighbors.NearestNeighbors(n_neighbors=len(self.points))
        nn.fit(self.points)
        self.indices = nn.kneighbors(self.points, return_distance=False)
        self.k = count
        self.d = dim

    def test_theta(self):
        """ Test the Θ-Graph
        """
        self.setup()
        np.random.seed(0)
        indices = ngl.theta_graph(self.points, self.k, 1, self.indices)[0]

        expected = set(range(1, self.k+1))
        actual = set(indices[indices != -1])

        msg = "\nNode {} Connectivity:".format(0)
        msg += "\n\texpected: {}\n\tactual: {} ".format(expected, actual)
        self.assertEqual(expected, actual, msg)

    def test_yao(self):
        """ Test the Yao Graph
        """
        self.setup()
        np.random.seed(0)
        indices = ngl.yao_graph(self.points, self.k, 1, self.indices)[0]

        expected = set(range(len(self.points)-1, self.k, -1))
        actual = set(indices[indices != -1])

        msg = "\nNode {} Connectivity:".format(0)
        msg += "\n\texpected: {}\n\tactual: {} ".format(expected, actual)
        self.assertEqual(expected, actual, msg)

    def test_ConeGraph_yao(self):
        """
        """
        self.setup()
        go = ngl.ConeGraph(num_sectors=self.k, algorithm="yao")
        np.random.seed(0)
        go.build(self.points)

        expected = set(range(len(self.points)-1, self.k, -1))
        actual = set()
        for (a, b, distance) in go:
            if a == 0:
                actual.add(b)
            elif b == 0:
                actual.add(a)

        msg = "\nNode {} Connectivity:".format(0)
        msg += "\n\texpected: {}\n\tactual: {} ".format(expected, actual)
        self.assertEqual(expected, actual, msg)

        # go = ngl.ConeGraph(num_sectors=self.k, algorithm="yao", query_size=2)
        # np.random.seed(0)
        # go.build(self.points)
        # actual = set()
        # for (a, b, distance) in go:
        #     if a == 0:
        #         actual.add(b)
        #     elif b == 0:
        #         actual.add(a)

        # msg = "\nNode {} Connectivity:".format(0)
        # msg += "\n\texpected: {}\n\tactual: {} ".format(expected, actual)
        # self.assertEqual(expected, actual, msg)

    def test_ConeGraph_theta(self):
        """
        """
        self.setup()

        go = ngl.ConeGraph(num_sectors=self.k, algorithm="theta")
        print(self.points[1:7])
        np.random.seed(0)
        go.build(self.points)

        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        lines = []
        for pt in self.points[1:]:
            lines.append([self.points[0], pt])

        plt.scatter(self.points[:,0], self.points[:, 1])
        for i in range(len(self.points)):
            plt.annotate(str(i), self.points[i])
        plt.gca().add_collection(LineCollection(lines))
        # plt.show()

        expected = set(range(1, self.k+1))
        actual = set()
        for (a, b, distance) in go:
            if a == 0:
                actual.add(b)
            elif b == 0:
                actual.add(a)

        msg = "\nNode {} Connectivity:".format(0)
        msg += "\n\texpected: {}\n\tactual: {} ".format(expected, actual)
        self.assertEqual(expected, actual, msg)

    def test_ConeGraph_invalid(self):
        """
        """
        self.setup()

        try:
            ngl.ConeGraph(num_sectors=self.k, algorithm="invalid")
            self.assertEqual(True, False, "A ValueError should be raised.")
        except ValueError as err:
            self.assertTrue(str(err).startswith("Unknown algorithm specified"))


if __name__ == "__main__":
    unittest.main()

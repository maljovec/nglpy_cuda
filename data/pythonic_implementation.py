#! /home/maljovec/anaconda3/bin/python
import numpy as np
import sklearn.neighbors
import time
import argparse
import sys

parser = argparse.ArgumentParser(
    description='Build an lp-beta skeleton using python.')
parser.add_argument('-b', dest='beta', type=float, default=1,
                    help='The value of beta')
parser.add_argument('-p', dest='p', type=float, default=2,
                    help='The value of the p-norm')
parser.add_argument('-i', dest='p_file', type=str, required=True,
                    help='The input data file as a csv.')
parser.add_argument('-n', dest='k_file', type=str, required=True,
                    help='The input neighborhood file.')
parser.add_argument('-a', dest='steepness', type=float, default=3,
                    help='The steepness parameter on the logistic probability function.')
parser.add_argument('-s', dest='steps', type=int, default=-1,
                    help='The number of steps to perform for discretization.')

# Accept, but ignore these parameters
parser.add_argument('-c', dest='count', type=int,
                    help='The number of points.')
parser.add_argument('-d', dest='dimensionality', type=int,
                    help='The number of dimensions.')
parser.add_argument('-k', dest='k', type=int,
                    help='The number of neighbors to be pruned.')

args = parser.parse_args()

start = time.time()
X = np.loadtxt(args.p_file)
problem_size = X.shape[0]
dimensionality = X.shape[1]
end = time.time()
print('Load data ({} s) shape={}'.format(end-start, X.shape), file=sys.stderr)

def logistic_function(x, r):
    k = args.steepness / r
    return 1 / (1 + np.exp(-k*(x - r)))


def paired_lpnorms(A, B, p=2):
    """ Method to compute the paired Lp-norms between two sets of points. Note,
    A and B should be the same shape.

    Args:
        A (MxN matrix): A collection of points
        B (MxN matrix): A collection of points
        p (positive float): The p value specifying what kind of Lp-norm to use
            to compute the shape of the lunes.
    """
    N = A.shape[0]
    dimensionality = A.shape[1]
    norms = np.zeros(N)
    for i in range(N):
        norm = 0.0
        for k in range(dimensionality):
            norm += (A[i, k] - B[i, k])**p
        norms[i] = norm**(1./p)
    return norms


def min_distance_from_edge(t, beta, p):
    """ Using a parameterized scale from [0,1], this function will determine
    the minimum valid distance to an edge given a specified lune shape defined
    by a beta parameter for defining the radius and a p parameter specifying
    the type of Lp-norm the lune's shape will be defined in.

    Args:
        t (float): the parameter value defining how far into the edge we are.
        0 means we are at one of the endpoints, 1 means we are at the edge's
        midpoint.
        beta (float): The beta value for the lune-based beta-skeleton
        p (float): The p value specifying which Lp-norm to use to compute
            the shape of the lunes. A negative value will be
            used to represent the inf norm

    """
    xC = 0
    yC = 0
    if t > 1:
        return 0
    if beta <= 1:
        r = 1 / beta
        yC = (r**p - 1)**(1. / p)
    else:
        r = beta
        xC = 1 - beta
    y = (r**p - (t-xC)**p)**(1. / p) - yC
    return 0.5*y


def create_template(beta, p=2, steps=100):
    """ Method for creating a template that can be mapped to each edge in
    a graph, since the template is symmetric, it will map from one endpoint
    to the center of the edge.

    Args:
        beta (float [0,1]): The beta value for the lune-based beta-skeleton
        p (positive float): The p value specifying which Lp-norm to use to
            compute the shape of the lunes.
    """
    template = np.zeros(steps+1)
    if p < 0:
        if beta >= 1:
            template[:-1] = beta/2
        return template
    for i in range(steps):
        template[i] = min_distance_from_edge(i/steps, beta, p)
    return template


start = time.time()
edges = np.loadtxt(args.k_file, dtype=int)
end = time.time()
print('Neighborhood graph loaded ({} s)'.format(end-start), file=sys.stderr)


def prune_discrete(X, edges, beta=1, lp=2, steps=99):
    # problem_size = min(10000, edges.shape[0]) # edges.shape[0]
    problem_size = edges.shape[0]
    template = create_template(beta, lp, steps)
    pruned_edges = np.zeros(shape=edges.shape) - 1
    for i in range(problem_size):
        # print(i,problem_size)
        p = X[i]
        # Xp = X - p
        for k in range(edges.shape[1]):
            ###################################################################
            #  0	2%
            j = edges[i, k]
            q = X[j]
            if i == j:
                continue
            ###################################################################
            #  1	1%
            pq = q - p
            ###################################################################
            #  2	8%
            edge_length = np.linalg.norm(pq)
            ###################################################################
            #  3	4%
            # adjacent_indices = []
            # for m in range(len(edges)):
            #     row = edges[m]
            #     if np.any(np.logical_or(row == i, row == j)):
            #         adjacent_indices.append(m)
            # subset = np.concatenate((edges[i], edges[j], np.array(adjacent_indices)))
            subset = np.concatenate((edges[i], edges[j]))
            ###################################################################
            #  4	21%
            # subset = np.unique(subset)
            ###################################################################
            #  5	11%
            Xp = X[subset] - p
            ###################################################################
            #  6	8%
            projections = np.dot(Xp, pq)/(edge_length**2)
            ###################################################################
            #  7	5%
            temp_indices_1 = projections * 2 - 1
            ###################################################################
            #  8	2%
            temp_indices_2 = steps*temp_indices_1
            ###################################################################
            #  9	2%
            temp_indices_3 = np.rint(temp_indices_2)
            ###################################################################
            # 10	1%
            temp_indices_4 = temp_indices_3.astype(np.int64)
            # timings[10] += time.time() - start
            ###################################################################
            # 11	2%
            lookup_indices = np.abs(temp_indices_4)
            ###################################################################
            # 12	3%
            temp_indices_5 = np.logical_and(
                lookup_indices >= 0, lookup_indices <= steps)
            ###################################################################
            # 13	3%
            valid_indices = np.nonzero(temp_indices_5)[0]
            ###################################################################
            # 14	9%
            temp = np.atleast_2d(projections[valid_indices]).T*pq
            ###################################################################
            # 15	8%
            distances_to_edge = paired_lpnorms(Xp[valid_indices], temp)
            ###################################################################
            # 16	7%
            points_in_region = np.nonzero(
                distances_to_edge < edge_length*template[lookup_indices[valid_indices]])[0]
            ###################################################################
            # 17	1%
            if len(points_in_region) == 0:
                pruned_edges[i, k] = j
    return pruned_edges


def prune(X, edges, beta=1, lp=2):
    # problem_size = min(10000, edges.shape[0]) # edges.shape[0]
    problem_size = edges.shape[0]
    pruned_edges = np.zeros(shape=edges.shape) - 1
    # timings = 14*[0]
    for i in range(problem_size):
        # print(i,problem_size)
        p = X[i]
        # Xp = X - p
        for k in range(edges.shape[1]):
            ###################################################################
            #  0
            j = edges[i, k]
            q = X[j]
            if i == j:
                continue
            ###################################################################
            #  1
            pq = q - p
            ###################################################################
            #  2
            edge_length = np.linalg.norm(pq)
            ###################################################################
            #  3
            # adjacent_indices = []
            # for m in range(len(edges)):
            #     row = edges[m]
            #     if np.any(np.logical_or(row == i, row == j)):
            #         adjacent_indices.append(m)
            # subset = np.concatenate((edges[i], edges[j],
            #                          np.array(adjacent_indices)))
            subset = np.concatenate((edges[i], edges[j]))
            ###################################################################
            #  4
            # subset = np.unique(subset)
            ###################################################################
            #  5
            Xp = X[subset] - p
            ###################################################################
            #  6
            projections = np.dot(Xp, pq)/(edge_length**2)
            ###################################################################
            #  7
            temp_indices = np.logical_and(projections > 0.,
                                          np.logical_and(projections < 1.,
                                                         np.logical_and(subset != i,
                                                                        subset != j)))
            ###################################################################
            #  8
            valid_indices = np.nonzero(temp_indices)[0]
            ###################################################################
            #  9
            temp = np.atleast_2d(projections[valid_indices]).T*pq
            ###################################################################
            # 10
            min_distances = np.zeros(len(valid_indices))
            for idx, t in enumerate(projections[valid_indices]):
                min_distances[idx] = min_distance_from_edge(
                    abs(2*t-1), beta, lp) * edge_length
            # timings[10] += time.time() - start
            ###################################################################
            # 11
            distances_to_edge = paired_lpnorms(Xp[valid_indices], temp)
            # timings[11] += time.time() - start
            ###################################################################
            # 12
            points_in_region = np.nonzero(distances_to_edge < min_distances)[0]
            # timings[12] += time.time() - start
            ###################################################################
            # 13
            if len(points_in_region) == 0:
                pruned_edges[i, k] = j
            # timings[13] += time.time() - start
    # return pruned_edges, timings
    return pruned_edges

def get_probability(X, edges, beta=1, lp=2):
    # problem_size = min(10000, edges.shape[0]) # edges.shape[0]
    problem_size = edges.shape[0]
    probabilities = np.ones(shape=edges.shape)
    for i in range(problem_size):
        # print(i,problem_size)
        p = X[i]
        # Xp = X - p
        for k in range(edges.shape[1]):
            ###################################################################
            #  0
            j = edges[i, k]
            q = X[j]
            if i == j:
                probabilities[i, k] = 0
                continue
            ###################################################################
            #  1
            pq = q - p
            ###################################################################
            #  2
            edge_length = np.linalg.norm(pq)
            ###################################################################
            #  3
            # adjacent_indices = []
            # for m in range(len(edges)):
            #     row = edges[m]
            #     if np.any(np.logical_or(row == i, row == j)):
            #         adjacent_indices.append(m)
            # subset = np.concatenate((edges[i], edges[j],
            #                          np.array(adjacent_indices)))
            subset = np.concatenate((edges[i], edges[j]))
            ###################################################################
            #  4
            # subset = np.unique(subset)
            ###################################################################
            #  5
            Xp = X[subset] - p
            ###################################################################
            #  6
            projections = np.dot(Xp, pq)/(edge_length**2)
            ###################################################################
            #  7
            temp_indices = np.logical_and(projections > 0.,
                                          np.logical_and(projections < 1.,
                                                         np.logical_and(subset != i,
                                                                        subset != j)))
            ###################################################################
            #  8
            valid_indices = np.nonzero(temp_indices)[0]
            ###################################################################
            #  9
            temp = np.atleast_2d(projections[valid_indices]).T*pq
            ###################################################################
            # 10
            min_distances = np.zeros(len(valid_indices))
            for idx, t in enumerate(projections[valid_indices]):
                min_distances[idx] = min_distance_from_edge(
                    abs(2*t-1), beta, lp) * edge_length
            ###################################################################
            # 11
            distances_to_edge = paired_lpnorms(Xp[valid_indices], temp)
            ###################################################################
            # 12
            probs = logistic_function(distances_to_edge, min_distances)

            print('P(pq), for p={}, q={}'.format(i, j))
            print('\tr p(pq)')
            for dd, dp in zip(valid_indices, probs):
                print('\t{:2d} {:3f}'.format(dd, dp))
            ###################################################################
            # 13
            if len(probs) > 0:
                probabilities[i, k] = np.min(probs)
    return probabilities


start = time.time()
if args.steps > 0:
    pruned_edges = prune_discrete(X, edges, beta=args.beta, lp=args.p,
                                  steps=args.steps)
else:
    pruned_edges = prune(X, edges, beta=args.beta, lp=args.p)

probabilities = get_probability(X, edges, beta=args.beta, lp=args.p)
end = time.time()
print('Actual prune function ({} s)'.format(end-start), file=sys.stderr)

print(pruned_edges)
print(probabilities)

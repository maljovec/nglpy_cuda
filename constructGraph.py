import numpy as np
from sklearn.neighbors import NearestNeighbors
import argparse
import sys

parser = argparse.ArgumentParser(description='Build a knn graph.')
parser.add_argument('-i', dest='p_file', type=str, required=True,
                    help='The input data file as a csv.')
parser.add_argument('-k', dest='k', type=int,  required=True,
                    help='The number of neighbors to be pruned.')
args = parser.parse_args()

# ['python', 'constructGraph.py', '-d', str(D), '-b', '1', '-k', str(K), '-i', p_file],
X = np.loadtxt(args.p_file)
nbrs = NearestNeighbors(n_neighbors=args.k).fit(X)
neighbors = nbrs.kneighbors(X, return_distance=False)
for i, row in enumerate(neighbors):
    for j in row:
        print('{} {}'.format(i, j))
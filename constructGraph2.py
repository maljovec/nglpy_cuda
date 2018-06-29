from pyflann import *
import numpy as np
import argparse
import sys

parser = argparse.ArgumentParser(description='Build a knn graph.')
parser.add_argument('-i', dest='p_file', type=str, required=True,
                    help='The input data file as a csv.')
parser.add_argument('-k', dest='k', type=int,  required=True,
                    help='The number of neighbors to be pruned.')
args = parser.parse_args()

X = np.loadtxt(args.p_file)

flann = FLANN()
neighbors, _ = flann.nn(X, X, 2, algorithm="kmeans",
                        branching=32, iterations=7, checks=16)
for row in neighbors:
    sep = ''
    for j in row:
        sys.stdout.write('{}{}'.format(sep, j))
        sep = ' '
    sys.stdout.write('\n')

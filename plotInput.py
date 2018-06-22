import matplotlib.pyplot as plt
from matplotlib import collections as mc
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Generating sample sets for testing.')
parser.add_argument('seed', type=int, help='The seed for the RNG.')
parser.add_argument('count', type=int, help='The number of points to generate.')
parser.add_argument('algorithm', type=str, help='The sampling algorithm to display')

args = parser.parse_args()

colors = { 'uniform': '#8da0cb',
           'normal': '#66c2a5',
           'lhs': '#e78ac3',
           'cvt': '#fc8d62'}

fig, ax = plt.subplots()
fname = '{}_{}_2_{}.csv'.format(args.algorithm, args.count, args.seed)
X = np.loadtxt('data/input/'+fname)
ax.scatter(X[:,0], X[:,1], s=1, c=colors[args.algorithm])

ax.autoscale()
ax.margins(0.1)
ax.set_xlim(0,1)
ax.set_ylim(0,1)
# plt.legend()
ax.set_title(fname)
plt.show()
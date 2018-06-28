#! /home/maljovec/anaconda3/bin/python
import numpy as np
import argparse
import pyDOE
import subprocess
import sys

parser = argparse.ArgumentParser(description='Generating sample sets for testing.')
parser.add_argument('dimensionality', type=int,
                    help='The dimensionality of the problem to handle.')
parser.add_argument('seed', type=int, help='The seed for the RNG.')
parser.add_argument('count', type=int, help='The number of points to generate.')
parser.add_argument('strategy', type=str, help='How the points should be sampled')
parser.add_argument('outfile', type=str, help='Where the points should be written')

args = parser.parse_args()
D = args.dimensionality
N = args.count

np.random.seed(args.seed)

if args.strategy == 'uniform':
    X = np.random.uniform(size=(N,D))
elif args.strategy == 'normal':
    X = np.clip(np.random.normal(loc=0.5, scale=0.1, size=(N,D)), 0, 1)
elif args.strategy == 'cvt':
    result = subprocess.run(['cvt/createCVT', '-N', '{}'.format(N), '-D',
                             '{}'.format(D), '-seed', '{}'.format(args.seed),
                             '-ann', '1', '-iterations', '1000000'],
                             stdout=subprocess.PIPE)
    lines = result.stdout.decode('utf-8').strip().split('\n')
    X = np.zeros((N,D))
    for i,line in enumerate(lines):
        X[i,:] = list(map(float, line.strip().split(' ')))
elif args.strategy == 'lhs':
    X = pyDOE.lhs(D, N)

fname = args.outfile
np.savetxt(fname, X)
sys.stderr.write(fname)
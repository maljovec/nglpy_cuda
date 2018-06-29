"""
    A flask application for visualizing the results of a 2D neighborhood
    graph
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
# import os
import subprocess
import json
import time

app = Flask(__name__)
# ######################################################################
# Constants
D = 2
sample_program = 'generate_sample_set.py'
knn_program = 'knn-generator/binsrc/getNeighborGraph'
skeleton_program = {'base': 'ngl-base/binsrc/getNeighborGraph',
                    'omp': 'ngl-omp/binsrc/getNeighborGraph',
                    'numba': 'ngl-numba/test_numba.py',
                    'gpu': 'ngl-gpu/prune_graph',
                    'numbad': 'ngl-numba/test_numba.py',
                    'gpud': 'ngl-gpu/prune_graph'}
# ######################################################################
# User-editable parameters
N = 10
seed = 0
s_type = 'uniform'
g_type = 'knn'
# ######################################################################
# Dynamically computed constants
p_file = 'data/input/points_{}_{}_{}_{}.csv'.format(s_type, N, D, seed)
k_file = 'data/graphs/knn_{}_{}_{}_{}.txt'.format(s_type, N, D, seed)
X = np.zeros((N, D))
# ######################################################################


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/sample", methods=['POST'])
def makeData():
    if request.method == 'POST':
        start = time.time()
        params = request.get_json()
        print(params)

        N = int(params['count'])
        seed = int(params['seed'])
        s_type = params['s_type']

        p_file = 'data/input/points_{}_{}_{}_{}.csv'.format(s_type, N, D,
                                                            seed)
        # if not (os.path.isfile(p_file) and not os.stat(e_file).st_size):
        if True:
            result = subprocess.run(['python', sample_program, str(D),
                                     str(seed), str(N), s_type, p_file],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, check=True)
            print(result.args)
        X = np.loadtxt(p_file)
        end = time.time()
        print(p_file)
        return jsonify({'data': json.dumps(X.tolist(), separators=(',', ':')),
                        'time': '{:6.4f} s'.format(end-start)})


@app.route("/graph", methods=['POST'])
def computeGraph():
    if request.method == 'POST':
        start = time.time()
        params = request.get_json()
        print(params)

        N = int(params['count'])
        seed = int(params['seed'])
        s_type = params['s_type']

        K = params['k']
        beta = params['beta']
        p = params['p']
        g_type = params['g_type']
        steps = params['steps']
        if steps > 0 and g_type in ['gpu', 'numba']:
            g_type += 'd'

        K = min(K, N)
        p_file = 'data/input/points_{}_{}_{}_{}.csv'.format(s_type, N, D,
                                                            seed)
        k_file = 'data/graphs/knn_{}_{}_{}_{}_{}.txt'.format(s_type, N, D,
                                                             K, seed)
        e_file = 'data/output/{}_{}_{}_{}_{}_{}.txt'.format(g_type, s_type,
                                                            N, D, K, seed)
        edge_set = set()
        # if not (os.path.isfile(e_file) and os.stat(e_file).st_size):
        if True:
            # if not (os.path.isfile(k_file) and os.stat(k_file).st_size):
            if True:
                fptr = open(k_file, 'w')
                result = subprocess.run([knn_program, '-d', str(D), '-b', '1',
                                         '-k', str(K), '-i', p_file],
                                        stdout=subprocess.PIPE, stderr=fptr,
                                        check=True)
                print(result.args)
                fptr.close()

            if g_type == 'knn':
                edges = np.atleast_2d(np.loadtxt(k_file, dtype=int))
                if (edges.shape[0] == 1):
                    edges = edges.T
                for i, row in enumerate(edges):
                    for j in row:
                        lo = min(i, j)
                        hi = max(i, j)
                        if lo != hi:
                            edge_set.add((lo, hi))
            else:
                cmd = ['/usr/bin/time', '-f', '%e',
                       skeleton_program[g_type], '-d', str(D), '-b',
                       str(beta),  '-k', str(K), '-i', p_file, '-n',
                       k_file, '-c', str(N), '-p', str(p), '-s',
                       str(steps)]
                fptr = open(e_file, 'w')
                result = subprocess.run(cmd, stdout=fptr,
                                        stderr=subprocess.PIPE)
                fptr.close()
                print(result.args)
                print(result.returncode, result.stderr.decode('utf-8'))

                fptr = open(e_file)
                for line in fptr:
                    edge = list(map(int, line.split(' ')))
                    lo = min(edge)
                    hi = max(edge)
                    if lo != hi:
                        edge_set.add((lo, hi))
                fptr.close()
        else:
            edges = np.atleast_2d(np.loadtxt(e_file, dtype=int))
            for edge in edges:
                lo = min(edge)
                hi = max(edge)
                if lo != hi:
                    edge_set.add((lo, hi))

        edges = ''
        for e in edge_set:
            edges += '{},{};'.format(e[0], e[1])
        end = time.time()
        print(e_file)
        return jsonify({'edges': edges,
                        'time': '{:6.4f} s'.format(end-start)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

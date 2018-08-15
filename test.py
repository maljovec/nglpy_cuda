import nglpy_cuda
import numpy as np
from colorama import init
from termcolor import colored


def test_min_distance_from_edge():
    print(colored('~'*80, 'yellow'))
    print(colored('Testing min_distance_from_edge', 'yellow'))
    assert 0.5 == nglpy_cuda.min_distance_from_edge(0, 1, 2)
    assert 0 == nglpy_cuda.min_distance_from_edge(1, 1, 2)
    print(colored('PASSED!', 'green'))

def test_create_template(beta, p, steps):
    print(colored('~'*80, 'yellow'))
    print(colored('Testing create_template', 'yellow'))
    template = nglpy_cuda.create_template(beta, p, steps)
    assert len(template) == 50
    for i in range(len(template)-1):
        assert template[i] > template[i+1]

def test_prune(N, D, k, p, beta, X, edges):
    print(colored('PASSED!', 'green'))
    print(colored('~'*80, 'yellow'))
    print(colored('Testing prune', 'yellow'))
    #print('python:')
    #print(colored(edges[:5,:], 'green'))
    #print('C++:')
    ngl_edges = nglpy_cuda.prune(N, D, k, p, beta, X, edges)
    print(ngl_edges)
    print(colored('PASSED!', 'green'))

def test_prune_discrete(N, D, k, steps, beta, p, X, edges):
    print(colored('~'*80, 'yellow'))
    print(colored('Testing prune_discrete with beta/lp specified', 'yellow'))
    ngl_edges = nglpy_cuda.prune_discrete(N, D, k, steps, beta, p, X, edges)
    print(ngl_edges)
    print(colored('PASSED!', 'green'))
    print(colored('~'*80, 'yellow'))
    print(colored('Testing prune_discrete with template specified', 'yellow'))
    template = np.array(nglpy_cuda.create_template(beta, p, steps), dtype=np.float32)
    ngl_edges = nglpy_cuda.prune_discrete(N, D, k, steps, template, X, edges)
    print(ngl_edges)
    print(colored('PASSED!', 'green'))
    print(colored('~'*80, 'yellow'))

def test_get_edge_list():
    print(colored('Testing get_edge_list', 'yellow'))
    print(colored('Not Implemented Yet!', 'red'))
    # nglpy_cuda.get_edge_list()


D=2
N=1000
A='uniform'
S=0
k=5
beta = 1
p = 2
steps = 50
p_file = '../ngl-gpu/data/input/points_{}_{}_{}_{}.csv'.format(A, N, D, S)
k_file = '../ngl-gpu/data/graphs/knn_{}_{}_{}_{}_{}.txt'.format(A, N, D, k, S)
X = np.loadtxt(p_file, dtype=np.float32)
edges = np.loadtxt(k_file, dtype=np.int32)

init()
# I just want to make sure I am using the python reference counters correctly
# by doing some memory debugging with objgraph
#import pdb; pdb.set_trace()
#import objgraph
#objgraph.show_growth(limit=10)   # Stop and show change
test_min_distance_from_edge()
#objgraph.show_growth(limit=10)   # Stop and show change
test_create_template(beta, p, steps)
#objgraph.show_growth(limit=10)   # Stop and show change
test_prune(N, D, k, p, beta, X, edges)
#objgraph.show_growth(limit=10)   # Stop and show change
test_prune_discrete(N, D, k, steps, beta, p, X, edges)
#objgraph.show_growth(limit=10)   # Stop and show change
test_get_edge_list()
#objgraph.show_growth(limit=10)   # Stop and show change



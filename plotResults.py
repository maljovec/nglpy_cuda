import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numpy as np
import sys

draw_circles = True
draw_edges = False
annotate = False

X = np.loadtxt('data/input/data_2_{}_0.csv'.format(sys.argv[1]))

draw = {'base': False,
        'omp': False,
        'numba': False,
        'gpu': False,
        'gpu_discrete': True}

algorithms = { 'base': '#6598D0',
                'omp': '#15737C',
              'numba': '#FED950',
                'gpu': '#77B717',
       'gpu_discrete': '#588a00'}

edges = {}
eset = {}
for alg in algorithms:
    if alg == 'base' or draw[alg]:
        edges[alg] = np.loadtxt('data/output/edges_2D_{}.txt'.format(alg), dtype=int)
        eset[alg] = set()
        
        for edge in edges[alg]:
            if edge[1] < edge[0]:
                lo = edge[1]
                hi = edge[0]
            else:
                lo = edge[0]
                hi = edge[1]
            
            if lo != hi:
                eset[alg].add((lo,hi))
        
        print('{:>12}: {}'.format(alg, len(eset[alg])))

eset_base = eset['base']
color_base = algorithms['base']
for alg in algorithms:
    if alg == 'base' or not draw[alg]:
        continue
    eset_test = eset[alg]
    v1 = len(eset_base.difference(eset_test))
    v2 = len(eset_test.difference(eset_base))
    if v1+v2 > 0:
        print('Difference base vs {0}:\n         base only {1}\n {0:>12} only {2}'.format(alg, v1, v2))
    else: 
        print('No Difference base vs {0}'.format(alg))

fig, ax = plt.subplots()

for alg, color in algorithms.items():
    if draw[alg]:
        if draw_edges:
            lines = []
            for edge in eset[alg]:
                lines.append([(X[edge[0], 0], X[edge[0], 1]), (X[edge[1], 0], X[edge[1], 1])])
            lc = mc.LineCollection(lines, colors=color, linewidths=1, linestyles='--')
            ax.add_collection(lc)

        if draw_circles:
            for edge in eset[alg].difference(eset_base):
                lo = edge[0]
                hi = edge[1]
                mdpt = (X[hi] + X[lo])/2.
                diameter = np.linalg.norm(X[hi] - X[lo])
                radius = diameter/2.

                empty_region = plt.Circle((mdpt[0], mdpt[1]), radius, color=color, alpha=0.5)
                ax.add_artist(empty_region)
                ax.plot(X[[lo, hi], 0], X[[lo, hi], 1], c=color, linewidth=2, label=alg)

            for edge in eset_base.difference(eset[alg]):
                lo = edge[0]
                hi = edge[1]
                mdpt = (X[hi] + X[lo])/2.
                diameter = np.linalg.norm(X[hi] - X[lo])
                radius = diameter/2.

                empty_region = plt.Circle((mdpt[0], mdpt[1]), radius, color=color_base, alpha=0.5)
                ax.add_artist(empty_region)
                ax.plot(X[[lo, hi], 0], X[[lo, hi], 1], c=color_base, linewidth=2, label='base')

# ax.scatter(X[:,0], X[:,1])
ax.scatter(X[:,0], X[:,1], s=1, c='#fa9fb5')
if annotate:
    for i in range(len(X)):
        ax.annotate(i, (X[i,0], X[i,1]))

ax.autoscale()
ax.margins(0.1)
ax.set_xlim(0,1)
ax.set_ylim(0,1)
# plt.legend()
plt.show()
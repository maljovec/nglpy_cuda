=====
nglpy_cuda
=====

.. badges

.. image:: https://img.shields.io/pypi/v/nglpy_cuda.svg
        :target: https://pypi.python.org/pypi/nglpy_cuda
        :alt: PyPi
.. image:: https://travis-ci.org/maljovec/nglpy_cuda.svg?branch=master
        :target: https://travis-ci.org/maljovec/nglpy_cuda
        :alt: Travis-CI
.. image:: https://coveralls.io/repos/github/maljovec/nglpy_cuda/badge.svg?branch=master
        :target: https://coveralls.io/github/maljovec/nglpy_cuda?branch=master
        :alt: Coveralls
.. image:: https://readthedocs.org/projects/nglpy_cuda/badge/?version=latest
        :target: https://nglpy_cuda.readthedocs.io/en/latest/?badge=latest
        :alt: ReadTheDocs
.. image:: https://pyup.io/repos/github/maljovec/nglpy_cuda/shield.svg
        :target: https://pyup.io/repos/github/maljovec/nglpy_cuda/
        :alt: Pyup

.. end_badges

.. logo

.. image:: docs/_static/nglpycu.svg
    :align: center
    :alt: nglpycu

.. end_logo

.. introduction

A reimplementation of the Neighborhood Graph Library
(NGL_) developed by Carlos Correa and Peter Lindstrom that
supports pruning a graph on the GPU. Developed as a
replacement for nglpy_ where a CUDA-compatible GPU is
available.

.. _NGL: http://www.ngraph.org/

.. _nglpy: https://github.com/maljovec/nglpy

.. LONG_DESCRIPTION

Given a set of arbitrarily arranged points in any dimension, this library is
able to construct several different types of neighborhood graphs mainly focusing
on empty region graph algorithms such as the beta skeleton family of graphs.

Consider using an optimized approximate nearest neighbor library (see ann-benchmarks_
for an updated list of algorithms and their relative performance) to construct the
initial graph to be pruned, otherwise this library will rely on the exact k-nearest
algorithm provided by scikit-learn_.

.. _ann-benchmarks: http://ann-benchmarks.com/

.. _scikit-learn: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors

.. END_LONG_DESCRIPTION

.. end_introduction

.. prerequisites

Prerequisites
=============

Nvidia CUDA Toolkit (TODO: determine minimum version number) - tested on 9.1.

Otherwise, all other python requirements can be installed via pip::

    pip install -r requirements.txt

.. end_prerequisites

.. install

.. Installation
.. ============
..
..
.. ::
..
..    pip install nglpy_cuda
..

.. end-install

.. build

Build
=====

Until I get this packaged appropriately, use the following command to compile the CUDA code::

    nvcc src/ngl_cuda.cu -I include/ --compiler-options "-fPIC" --shared -o libnglcu.so

The CUDA API can then be tested with a small C++ example (TODO: provide small data file in repo for testing this next line)::

    g++ -L. -I include/ src/test.cpp -lnglcu -o test
    ./test -i <input file> -d <# of dimensions> -c <# of points> -n <neighbor edge file> -k <k neighbors to prune> -b <beta parameter> -p <shape descriptor> -s <discretization steps>

For now, don't install this yet, but set it up in development mode::

    python setup.py develop

Run the test suite to verify it is able to make the CUDA calls without erroring::

    python setup.py test

.. end_build

.. usage

Usage
=====

TODO

.. end-usage

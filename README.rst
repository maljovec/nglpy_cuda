==========
nglpy-cuda
==========


.. image:: https://img.shields.io/pypi/v/nglpy_cuda.svg
        :target: https://pypi.python.org/pypi/nglpy_cuda

.. image:: https://img.shields.io/travis/maljovec/nglpy_cuda.svg
        :target: https://travis-ci.org/maljovec/nglpy_cuda

.. image:: https://readthedocs.org/projects/nglpy-cuda/badge/?version=latest
        :target: https://nglpy-cuda.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/maljovec/nglpy_cuda/shield.svg
     :target: https://pyup.io/repos/github/maljovec/nglpy_cuda/
     :alt: Updates



TODO


* Free software: BSD license
* Documentation: https://nglpy-cuda.readthedocs.io.

Prerequisite
--------
Nvidia CUDA Toolkit (TODO: determine minimum version number) - tested on 9.1

Build
--------

Until I get this packaged appropriately, use the following command to compile the CUDA code:

```bash
nvcc src/ngl_cuda.cu -I include/ --compiler-options "-fPIC" --shared -o libnglcu.so
```

The CUDA API can then be tested with a small C++ example (TODO: provide small data file in repo for testing this next line):
```bash
g++ -L. -I ../include/ ../src/test.cpp -lnglcu -o test
./test -i <input file> -d <# of dimensions> -c <# of points> -n <neighbor edge file> -k <k neighbors to prune> -b <beta parameter> -p <shape descriptor> -s <discretization steps>
```

For now, don't install this yet, but set it up in development mode:

```bash
python setup.py develop
```

Run the small test script to verify it is able to make the CUDA calls without erroring:

```bash
python test.py
```

Features
--------

* TODO

Known Issues
--------
* No support yet for relaxed Graphs
* CUDA/C code only deals with single precision floating point, requiring the user to convert their numpy arrays, either allow for either-or or write python-side wrapper to abstract this away from the user.
* Have nvcc run in python setup.py to remove extraneous step.
* Add point and neighborhood files for C++ example
* Documentation
* Continuous integration
* Create template returns an array that is incompatible with the ```prune_discrete``` function. This should return a numpy array

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

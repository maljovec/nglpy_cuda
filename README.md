# nglpy-cuda
[![PyPI](https://img.shields.io/pypi/v/nglpy_cuda.svg)](https://pypi.python.org/pypi/nglpy_cuda)
[![TravisCI](https://img.shields.io/travis/maljovec/nglpy_cuda.svg)](https://travis-ci.org/maljovec/nglpy_cuda)
[![Coverage Status](https://coveralls.io/repos/github/maljovec/nglpy_cuda/badge.svg?branch=master)](https://coveralls.io/github/maljovec/nglpy_cuda?branch=master)
[![ReadTheDocs](https://readthedocs.org/projects/nglpy-cuda/badge/?version=latest)](https://nglpy-cuda.readthedocs.io/en/latest/?badge=latest)
[![Pyup](https://pyup.io/repos/github/maljovec/nglpy_cuda/shield.svg)](https://pyup.io/repos/github/maljovec/nglpy_cuda/)

TODO: Description

* Free software: BSD license
* Documentation: https://nglpy-cuda.readthedocs.io.

# Prerequisites

Nvidia CUDA Toolkit (TODO: determine minimum version number) - tested on 9.1

```bash
pip install -r requirements.txt
```

# Build

Until I get this packaged appropriately, use the following command to compile the CUDA code:
```bash
nvcc src/ngl_cuda.cu -I include/ --compiler-options "-fPIC" --shared -o libnglcu.so
```

The CUDA API can then be tested with a small C++ example (TODO: provide small data file in repo for testing this next line):
```bash
g++ -L. -I include/ src/test.cpp -lnglcu -o test
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

# Features

* TODO

# Known Issues
* Python calls do not appear to be working correctly
* No support yet for relaxed Graphs
* CUDA/C code only deals with single precision floating point, requiring the user to convert their numpy arrays, either allow for both double and float types or write python-side wrapper to abstract this away from the user and force their data into single-precision.
* Have nvcc run in python setup.py to remove extraneous step.
* Add point and neighborhood files for C++ example
* Documentation
* ```create_template``` returns an array that is incompatible with the ```prune_discrete``` function. This should return a numpy array
* Formalize test framework
* Setup code coverage for CUDA/C++ code
* Setup continuous integration
* Get all of the badges on this page working correctly: pyup, travis, readthedocs, pypi.

# Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template.

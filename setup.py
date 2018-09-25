#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup

########################################################################
# Code from https://github.com/rmcgibbo/npcuda-example to build a custom
# CUDA module via distutils
import os
import re
from os.path import join as pjoin
from distutils.extension import Extension
from distutils.command.build_ext import build_ext
import numpy

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['numpy', 'scipy', 'sklearn']


def get_property(prop, project):
    """
        Helper function for retrieving properties from a project's
        __init__.py file
        @In, prop, string representing the property to be retrieved
        @In, project, string representing the project from which we will
        retrieve the property
        @Out, string, the value of the found property
    """
    result = re.search(
        r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
        open(project + "/__init__.py").read(),
    )
    return result.group(1)


def find_in_path(name, path):
    "Find a file in a search path"
    # adapted fom https://bit.ly/2QEHMUt
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                                   'located in your $PATH. Either add '
                                   'it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home,
                  'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}

    if not os.path.exists(cudaconfig['lib64']):
        cudaconfig['lib64'] = pjoin(home, 'lib')

    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError(
                'The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig


CUDA = locate_cuda()


def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile

# run the customize_compiler


class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


VERSION = get_property("__version__", "nglpy_cuda")
setup_requirements = ['numpy']
test_requirements = []

nglpy_cuda_core = Extension('nglpy_cuda.core',
                            sources=['nglpy_cuda/core.cpp', 'src/ngl_cuda.cu'],
                            include_dirs=['include', CUDA['include'],
                                          numpy.get_include()],
                            library_dirs=[CUDA['lib64']],
                            runtime_library_dirs=[CUDA['lib64']],
                            libraries=['cudart'],
                            extra_compile_args={'gcc': [],
                                                'nvcc': ['-c',
                                                         '--compiler-options',
                                                         "'-fPIC'"]})

setup(
    author="Daniel Patrick Maljovec",
    author_email='maljovec002@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="TODO",
    install_requires=requirements,
    license="BSD license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='nglpy_cuda',
    name='nglpy_cuda',
    setup_requires=setup_requirements,
    test_suite='nglpy_cuda.tests',
    # tests_require=test_requirements,
    url='https://github.com/maljovec/nglpy_cuda',
    version=VERSION,
    zip_safe=False,
    ext_modules=[nglpy_cuda_core],
    cmdclass={'build_ext': custom_build_ext},
    packages=['nglpy_cuda']
)

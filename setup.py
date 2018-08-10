#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages, Extension

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = []

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

ngl_cuda_core = Extension('ngl_cuda.core', sources=['src/core.cpp'],
                          include_dirs=['include'], libraries=['ngl_cuda'])

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
    packages=find_packages(include=['nglpy_cuda']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/maljovec/nglpy_cuda',
    version='0.1.0',
    zip_safe=False,
    ext_modules=[],
    packages=['ngl_cuda']
)

import os
import sys
import numpy as np
from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy as np
from path import Path

include_gsl_dir = '/usr/local/include/'
lib_gsl_dir = '/usr/local/lib/'

extension_names = ["bessel", "dncbtd", "mcmc_model_parallel","prbf","prbgnmf", "sample"]

extensions = [
    Extension(f"{x}", [f"./dncbfac/cython_files/{x}.pyx"],
              library_dirs=[lib_gsl_dir],
              libraries=['gsl', 'gslcblas'],
              extra_compile_args=['-fopenmp'],
              extra_link_args=['-fopenmp'],
              include_dirs=[include_gsl_dir, np.get_include()])
    for x in extension_names ]

setup(
    name='dncbfac',
    version='1.0.0',
  	cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(extensions),
    packages=find_packages('dncbfac'),
    package_dir={'': 'dncbfac'},
    install_requires=[
        'click',
    ],
    entry_points={
        'console_scripts': [
            'dncbfac = dncbfac.cli:main',
        ],
    },
)

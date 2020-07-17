from setuptools import setup, find_packages
import sys, os.path

__version__ = '0.3.1'

if sys.version_info < (3,6):
    sys.exit('DeepDataProfiler requires Python 3.6 o later')

setup(
    name='deep_data_profiler',
    packages=['deep_data_profiler',
              'deep_data_profiler.classes',
              'deep_data_profiler.utils'],
    version=__version__,
    description='DNN Data Profiling is a project based on arXiv:1904.08089',
    license='3-Clause BSD license',
    long_description='''
    The DNN Data Profiling library provides tools for analyzing the internal decision structure of a deep neural network within the
    context of a specific dataset. The library was inspired by the work of Qiu, et al. in Adversarial Defense Through
    Network Profiling Based Path Extraction (2019), arXiv:1904.08089.

    The current version is preliminary. We are actively testing and would be grateful
    for comments and suggestions. Expect changes in both class names and methods as
    many of the requirements demanded of the library are worked out.
    ''',
    install_requires=['torch>=0.4.1','numpy>=1.14.3'],
    extras_require={
        'testing':['pytest>=4.0'],
        'documentation':['sphinx>=1.8.2','nb2plots>=0.6','sphinx-rtd-theme>=0.4.2'],
        'all':['sphinx>=1.8.2','nb2plots>=0.6','sphinx-rtd-theme>=0.4.2','pytest>=4.0',
                'torch>=0.4.1','pillow>=5.2.0','torchvision>=0.2.1','jupyter>=1.0.0','numpy>=1.14.3'],
    }
)

### Since this package is still in development, please install in a virtualenv or conda environment.
### See README for installations instructions



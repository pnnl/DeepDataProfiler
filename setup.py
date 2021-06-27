from setuptools import setup, find_packages
import sys
import os.path

__version__ = '1.0'

if sys.version_info < (3, 7):
    sys.exit('DeepDataProfiler requires Python 3.7 or later')

setup(
    name='deep_data_profiler',
    packages=['deep_data_profiler',
              'deep_data_profiler.classes',
              'deep_data_profiler.utils',
              'deep_data_profiler.algorithms',
              'deep_data_profiler.models'],
    version=__version__,
    description='DNN Data Profiling is a project based on arXiv:1904.08089',
    license='3-Clause BSD license',
    long_description='''
    The DNN Data Profiling library provides tools for analyzing the internal decision structure of a deep neural network within the
    context of a specific dataset. The library was inspired by the work of Qiu, et al. in Adversarial Defense Through
    Network Profiling Based Path Extraction (2019), arXiv:1904.08089. Full documentation may be found at https://pnnl.github.io/DeepDataProfiler/
    ''',
    install_requires=['torch>=1.3.0', 'numpy>=1.14.3', 'networkx>=2.5', 'scipy', 'ripser>=0.6.0', 'powerlaw>=1.4.6'],
    extras_require={
        'testing': ['pytest>=4.0'],
        'documentation': ['sphinx>=1.8.2', 'nb2plots>=0.6', 'sphinx-rtd-theme>=0.4.2'],
        'all': ['sphinx>=1.8.2', 'nb2plots>=0.6', 'sphinx-rtd-theme>=0.4.2', 'pytest>=4.0',
                'pillow>=5.2.0', 'torchvision>=0.4.1', 'jupyter>=1.0.0', 'opencv-python', 'pytorchcv'],
    }
)

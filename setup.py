from setuptools import setup, find_packages
import sys
import os.path

__version__ = "1.2.1"

if sys.version_info < (3, 7):
    sys.exit("DeepDataProfiler requires Python 3.7 or later")

setup(
    name="deep_data_profiler",
    packages=[
        "deep_data_profiler",
        "deep_data_profiler.classes",
        "deep_data_profiler.utils",
        "deep_data_profiler.algorithms",
        "deep_data_profiler.optimization",
    ],
    version=__version__,
    description="DNN Data Profiling is a project based on arXiv:1904.08089",
    license="3-Clause BSD license",
    long_description="""
    The Deep Data Profiling (DDP) library provides tools for analyzing the internal decision structure of a trained deep neural network.
    The library was inspired by the work of Qiu, et al. in Adversarial Defense Through
    Network Profiling Based Path Extraction (2019), arXiv:1904.08089.
    Full documentation may be found at https://pnnl.github.io/DeepDataProfiler/

    DDP contains code for generating graphical representations and feature visualizations,
    for VGG-like sequential and ResNet models implemented in PyTorch.
    DDP also provides tools from topological data analysis for analysis of these representations.

    Notice
    ------
    The research used in this repository is part of the Mathematics of Artificial Reasoning in Science (MARS) Initiative at Pacific Northwest National Laboratory (PNNL).  It was conducted under the Laboratory Directed Research and Development Program at PNNL, a multiprogram national laboratory operated by Battelle for the U.S. Department of Energy.

    Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.
    Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

    ```
             PACIFIC NORTHWEST NATIONAL LABORATORY
                      operated by
                       BATTELLE
                       for the
               UNITED STATES DEPARTMENT OF ENERGY
                under Contract DE-AC05-76RL01830
    ```
    """,
    install_requires=[
        "torch>=1.10.2",
        "numpy>=1.14.3",
        "networkx>=2.5",
        "scipy",
        "ripser>=0.6.0",
        "powerlaw>=1.4.6",
        "torchvision>=0.11.3",
    ],
    extras_require={
        "testing": ["pytest>=4.0"],
        "documentation": ["sphinx>=1.8.2", "nb2plots>=0.6", "sphinx-rtd-theme>=0.4.2"],
        "frontend": ["torch-lucent", "streamlit", "boto3"],
        "all": [
            "sphinx>=1.8.2",
            "nb2plots>=0.6",
            "sphinx-rtd-theme>=0.4.2",
            "pytest>=4.0",
            "pillow>=5.2.0",
            "torchvision>=0.4.1",
            "jupyter>=1.0.0",
            "opencv-python",
            "pytorchcv",
            "torch-lucent",
            "streamlit",
            "boto3",
        ],
    },
)

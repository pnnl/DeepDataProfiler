Deep Data Profiler (DDP)
========================

The DDP library provides tools for analyzing the internal decision structure of a deep neural network within the
context of a specific dataset. Documentation may be found [here](https://pnnl.github.io/DeepDataProfiler/)

Installing Deep Data Profiler
=============================

To install from PyPI:

    >>> pip install deep-data-profiler

Using Anaconda and Pip
----------------------

To install with tutorials and documentation using anaconda and pip:

Note: Documentation is built using Sphinx, which requires we install the matplotlib package using conda *not pip*.

    >>> conda create -n ddp python=3.7 matplotlib
    >>> conda activate ddp

From the root directory of deep_data_profiler do

    >>> pip install -e.[‘all’]

Then run

    >>> pytest

To see the sphinx documentation

    >>> open docs/index.html

You may also install without editing, tutorials and testing simply using

    >>> conda create -n ddp python=3.7
    >>> pip install .

Tutorials may be run in your browser using Google Colab
-------------------------------------------------------

<a href="https://colab.research.google.com/github/pnnl/DeepDataProfiler/blob/master/tutorials/Tutorial%201%20-%20DDP-Tutorial.ipynb" target="_blank">

  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    <span style={'margin-left':5px}>Tutorial 1 - An Introduction to a DDP Profile</span>
</a>
</br>

<a href="https://colab.research.google.com/github/pnnl/DeepDataProfiler/blob/master/tutorials/Tutorial%202%20-%20DDPAlgorithms.ipynb" target="_blank">

  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    <span style={'margin-left':5px}>Tutorial 2 - Topological Data Analysis</span>
</a>
</br>

<a href="https://colab.research.google.com/github/pnnl/DeepDataProfiler/blob/master/tutorials/Tutorial%203%20-%20SpectralAnalysis.ipynb" target="_blank">

  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    <span style={'margin-left':5px}>Tutorial 3 - Spectral Analysis of CNN</span>
</a>
</br>

A Streamlit visualization frontend
-------------------------------------------------------

[A visualization demo can be found here.](https://deepdataprofilerteam-deepdataprof-frontendmain-streamlit-vwolde.streamlit.app) The frontend tool is built with Streamlit, and showcases a few facets of the DDP library.

Notes
-----
The DDP project is part of the Mathematics of Artificial Reasoning in Science (MARS)
Initiative at Pacific Northwest National Laboratory (PNNL).  
Research was conducted under the Laboratory Directed Research and Development Program at PNNL,
a multiprogram national laboratory operated by Battelle for the U.S. Department of Energy.

* Principle Investigator: Brenda Praggastis
* Design and Development: Davis Brown, Brenda Praggastis, Madelyn Shapiro
* Topological Data Analysis Contributors: Emilie Purvine, Bei Wang
* Original authors: Nichole Nichols, Brenda Praggastis, Aaron Tuor

For questions and comments you may contact the developers directly at:
    deepdataprofiler@pnnl.gov

Notice
------
The research described in this work is part of the Mathematics of Artificial Reasoning in Science (MARS) Initiative at Pacific Northwest National Laboratory (PNNL).  It was conducted under the Laboratory Directed Research and Development Program at PNNL, a multiprogram national laboratory operated by Battelle for the U.S. Department of Energy.

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

License
-------

Released under the 3-Clause BSD license (see License.rst)

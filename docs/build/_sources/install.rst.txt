Installing Deep Data Profiler
=============================

Using Anaconda and Pip
----------------------

To install with tutorials and documentation using anaconda and pip:

Note: Documentation is built using Sphinx, which requires we install the matplotlib package using conda *not pip*. 
Also we must remove the mkl package automatically installed with the package to avoid having the kernel die on us.

	>>> conda create -n ddp python=3.9 matplotlib
	>>> conda activate ddp

From the root directory of deep_data_profiler do 

	>>> pip install -e.[‘all’]

Then run

	>>> pytest

To see the sphinx documentation 

	>>> open docs/build/index.htm

You may also install without editing, tutorials and testing simply using 

	>>> conda create -n ddp python=3.9
	>>> pip install .


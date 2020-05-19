========================
Deep Data Profiler (DDP)
========================

Description
===========
The Deep Data Profiler library provides tools for analyzing the internal decision structure of a deep neural network within the
context of a specific dataset. The library was inspired by the work of
Qiu, et al. in Adversarial Defense Through
Network Profiling Based Path Extraction (2019), arXiv:1904.08089.

The current version is preliminary. We are actively testing and would be
grateful
for comments and suggestions. Expect changes in both class names and
methods as
many of the requirements demanded of the library are worked out.

Notes
=====
Current ddp.classes package contains `vgg_profile.py`.
The code in this module is for reference only and is not being maintained.
User should use `ddp.classes.nn_profile.py` instead.


Contents
========

.. toctree::
   :maxdepth: 2

   overview/index
   Profiling <classes/modules.rst>
   Utilities <utils/modules.rst>
   install
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


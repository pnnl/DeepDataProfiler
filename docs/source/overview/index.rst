.. overview:

.. image:: /images/eaglegraph.png
    :width: 500
    :align: right
========
Overview
========

Introduction
------------

Deep neural networks (DNN) provide powerful and succesful classification tools.
Unfortunately they are often very complex and can involve millions of trained parameters.
This complexity makes it difficult to connect the input domain to the decisions used to classify its elements.
*Does the network really see a bird or does it just see blue sky and infer there is a bird?*
The `DeepDataProfiler`_ (DDP) challenge is to link hidden representations of existing trained neural networks to human recognizable input features
and to identify training invariant structures and metrics for determining trustworthiness of the network.

DDP is both a methodology and software package.
The methodology has two steps.

#. **Profiling:** Generate representations of data using the graph (or subnetwork) of neurons and synapses they most activate in a trained DNN.
#. **Analysis:** Apply Graph Theory and Topological Data Analysis (TDA) to identify structural features of the graphs that may be used to explain and interpret the decision process of the network.

The DDP software is the implemention of this methodology in Python using the PyTorch library.
Our first release in May 2020 profiled VGG-like Sequential CNN architectures.
The current release extends to ResNet architectures.
At present only models built using PyTorch can be profiled.
Our TDA work is implemented in the homology module in the algorithms directory and depends on ripser.

Related Work
------------

Providing human interpretable explanations for the decisions made by deep neural networks is a high
priority for machine learning practioners. One popular approach is to link input data to the most influential
**neurons** and **synapses** used for their classification.
Our research in this area was inspired by `Yuxian Qiu, Jingwen Leng, Cong Guo, Quan Chen, Chao Li,Minyi Guo, and Yuhao Zhu.
Adversarial Defense Through Net-work Profiling Based Path Extraction.
In2019 IEEE/CVF Con-ference on Computer Vision and Pattern Recognition (CVPR),
pages 4772–4781, Long Beach, CA, USA, June 2019. IEEE <https://arxiv.org/abs/1904.08089>`_
They create *graph-like* objects called **effective paths** to explore the internal organization of deep neural networks.

Deep neural networks lend themselves naturally to decomposition into graphs as they are already *graph like*
in their neural connections. Such decompositions have been explored in particular in image classification networks.
`Summit <https://fredhohman.com/summit/>`_ creates an **attribution graph** of the convolutional layers
to visualize neuron activations and linking, see
`F. Hohman, et al., “Summit: Scaling Deep Learning
Interpretability by Visualizing Activation and Attribution Summarizations,”  arXiv:1708.01785 [cs], Nov. 2017
<https://arxiv.org/abs/1904.02323>`_

Q. Zhang et al. create **explanatory graphs** to expose the knowledge hierarchy hidden inside a pre-trained
CNN, see: `Q. Zhang, R. Cao, F. Shi, Y. N. Wu, and S.-C. Zhu, “Interpreting CNN Knowledge via an Explanatory Graph,”
arXiv:1708.01785 [cs], Nov. 2017. <http://arxiv.org/abs/1708.01785>`_

N. Cammarata, et al. describe **circuits** of neurons from the convolutional layers of a network,
which connect to represent features
of an image used for classification, see `N. Cammarata, S. Carter, G. Goh, C. Olah, M. Petrov, and L. Schubert,
“Thread: Circuits,” Distill, vol. 5, no. 3, p. e24, Mar. 2020, doi: 10.23915/distill.00024.
<https://distill.pub/2020/circuits/>`_

.. image:: /images/Background.png
    :align: right

The `DeepDataProfiler`_  library builds on the ideas above to provide tools for the analysis of the internal decision
structure of a deep neural network.
Given a trained model, DDP generates **profile graphs** for inputs. These graphs are similar to the effective paths in `Qiu's <https://arxiv.org/abs/1904.08089>`_ paper
but with weighting and attribution similar to the graphs generated in `Summit <https://fredhohman.com/summit/>`_ .
At present the library has a
working pipeline to generate profiles using VGG and ResNet architectures implemented in `PyTorch <https://pytorch.org/>`_.

.. image:: /images/pipelineimg.png
    :align: right

We apply Graph Theory and Topological Data Analysis to the profile graphs to explore their structure.
The Algorithms directory contains modules
for exploring the persistent homology of point clouds
generated from profile graphs and for
exploring the empirical spectral decomposition
of the model's linear operators.

Jupyter notebooks for the library with illustrative examples are available in the tutorials directory.


.. toctree::
   :maxdepth: 2

.. _DeepDataProfiler: https://github.com/pnnl/DeepDataProfiler

CSBDeep – a toolbox for CARE
============================

.. image:: https://badge.fury.io/py/csbdeep.svg
   :target: https://pypi.org/project/csbdeep
   :alt: PyPI version

.. image:: https://github.com/CSBDeep/CSBDeep/workflows/Test/badge.svg
   :target: https://github.com/CSBDeep/CSBDeep/actions?query=workflow%3ATest
   :alt: Test

.. image:: https://github.com/CSBDeep/CSBDeep/workflows/Test%20(PyPI)/badge.svg
   :target: https://github.com/CSBDeep/CSBDeep/actions?query=workflow%3A%22Test+%28PyPI%29%22
   :alt: Test (PyPI)

.. image:: https://github.com/CSBDeep/CSBDeep/workflows/Test%20(Legacy)/badge.svg
   :target: https://github.com/CSBDeep/CSBDeep/actions?query=workflow%3A%22Test+%28Legacy%29%22
   :alt: Test (Legacy)

This is the documentation for the
`CSBDeep Python package <https://github.com/csbdeep/csbdeep>`_,
which provides a toolbox for content-aware restoration (CARE)
of (fluorescence) microscopy images, based on deep learning via
`Keras <https://keras.io/>`_ and `TensorFlow <https://www.tensorflow.org/>`_.
Please see the
`CSBDeep website <http://csbdeep.bioimagecomputing.com/>`_
for more information with links to our manuscript and supplementary material.
If you use this software for your research, please cite:

  `Content-Aware Image Restoration: Pushing the Limits of Fluorescence Microscopy <https://doi.org/10.1038/s41592-018-0216-7>`_.
  Martin Weigert, Uwe Schmidt, Tobias Boothe, Andreas Müller, Alexandr Dibrov, Akanksha Jain, Benjamin Wilhelm, Deborah Schmidt, Coleman Broaddus, Siân Culley, Mauricio Rocha-Martins, Fabián Segovia-Miranda, Caren Norden, Ricardo Henriques, Marino Zerial, Michele Solimena, Jochen Rink, Pavel Tomancak, Loic Royer, Florian Jug, and Eugene W. Myers.
  *Nature Methods 15.12 (2018): 1090–1097.*

After :doc:`installation </install>` of the Python package,
we recommend to follow the provided `examples`_
that provide step-by-step instructions via `Jupyter <http://jupyter.org/>`_ notebooks
on how to use this package.

.. _`examples`: http://csbdeep.bioimagecomputing.com/examples

.. note::
    This is an early version of the software.
    Several features are still missing, including the use of network
    ensembles and creating training data via simulation.


Table of contents
-----------------

.. toctree::
   install
   models
   datagen
   training
   prediction


.. **Last updated:** |today|

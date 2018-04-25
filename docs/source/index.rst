CSBDeep – a toolbox for applying CARE
=====================================

This is the documentation for the
`CSBDeep Python package <https://github.com/mpicbg-csbd/CSBDeep_code>`_
that allows to train and apply CARE networks. Please see the
`CSBDeep webpage <http://csbdeep.bioimagecomputing.com/>`_
for more information with links to our manuscript and supplementary material.

.. note::
    This is an early version of the software.
    It currently only supports CARE networks for direct image
    restoration/enhancement and isotropic reconstruction, i.e. surface
    extraction/projection is not available yet. Furthermore, the necessary
    training data can currently only be generated from aligned raw images (e.g. low- and
    high-SNR) or for isotropic reconstruction. Creating training data via
    simulation is currently not implemented.


After :doc:`installation </install>` of the Python package,
we recommend to follow the provided `Jupyter <http://jupyter.org/>`_ notebooks
that provide step-by-step instructions on how to use this package.
They can be found in the subfolder ``notebooks``:

#. `datagen.ipynb`_:
   Creating training data from aligned raw images.
   More documentation available at :doc:`datagen`.

#. `training.ipynb`_:
   Defining a CARE network and training it based on the data created in the first step.
   More documentation available at :doc:`training`.

#. `prediction.ipynb`_:
   Loading a trained CARE network and applying it to a raw image.
   More documentation available at :doc:`predict`.

Notebooks that demonstrate CARE networks for isotropic reconstruction
can be found in ``notebooks/isotropic``
(cf. `repository <https://github.com/mpicbg-csbd/CSBDeep_code/blob/master/notebooks/isotropic>`_).

.. _`datagen.ipynb`: https://github.com/mpicbg-csbd/CSBDeep_code/blob/master/notebooks/datagen.ipynb
.. _`training.ipynb`: https://github.com/mpicbg-csbd/CSBDeep_code/blob/master/notebooks/training.ipynb
.. _`prediction.ipynb`: https://github.com/mpicbg-csbd/CSBDeep_code/blob/master/notebooks/prediction.ipynb

Table of contents
-----------------

.. toctree::
   .. :maxdepth: 2

   install
   datagen
   training
   predict
   .. utils

.. .. image:: https://memegenerator.net/img/instances/81561610/work-in-progress.jpg
.. .. image::  https://memegenerator.net/img/instances/81561599/if-you-could-improve-this-that-would-be-great.jpg

.. .. todo::
..     Make documentation more like a tutorial

**Last updated:** |today|



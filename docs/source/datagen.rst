Generating training data
========================

In order to create the necessary data for training CARE networks, we
first need to specify matching pairs of raw source and target images.
To that end, we provide the function :func:`csbdeep.data.get_tiff_pairs_from_folders`,
which expects TIFF files with the same name, but in different folders.

.. note::
    - The raw data should be representative of all images that the CARE network
      will potentially be applied to after training.
    - Source and target images must be well-aligned to obtain effective CARE networks.

With the raw data specified as above, the function :func:`csbdeep.data.create_patches`
can be used to randomly extract patches of a given size that are suitable for training.
By default, patches are normalized based on a range of percentiles
computed on the raw images, which tends to lead to more robust CARE networks in our experience.
If not specified otherwise, patches which are purely background are also excluded
from being extracted, since they do not contain interesting structures.

.. todo::
   Training data generation like this will not work for:
       - surface projection → accommodate in :func:`csbdeep.data.create_patches`

.. autofunction:: csbdeep.data.get_tiff_pairs_from_folders
.. autofunction:: csbdeep.data.create_patches

.. autofunction:: csbdeep.data.no_background_patches
.. autofunction:: csbdeep.data.norm_percentiles
.. autofunction:: csbdeep.data.sample_percentiles

Isotropic reconstruction
------------------------

We provided the function :func:`csbdeep.data.anisotropic_distortions`
that returns a :obj:`csbdeep.data.Transform` object to be used
for creating training data for isotropic reconstruction networks.
See `Data augmention`_ to learn about transforms.

.. autofunction:: csbdeep.data.anisotropic_distortions

Advanced topics
---------------

.. todo::
    - Normalization
    - Patch filter

Custom data loaders
+++++++++++++++++++

If you cannot or do not want to put your raw images in a folder structure
as required by :func:`csbdeep.data.get_tiff_pairs_from_folders`, you
can create your custom :obj:`csbdeep.data.RawData` object to return
corresponding raw images in the required format for :func:`csbdeep.data.create_patches`.

.. autoclass:: csbdeep.data.RawData
    :members:

Data augmention
+++++++++++++++

Instead of recording raw images where structures of interest appear in all
possible appearance variations, it can be easier to augment the raw dataset
by including some of those variations that can be easily synthesized. Typical examples are axis-aligned rotations if structures of interest can appear at arbitrary rotations.
To that end, a :obj:`csbdeep.data.Transform` object can be used to specify such transformations
to augment the raw dataset.
We currently haven't implemented any transformations, but plan to at least
add axis-aligned rotations and flips later.

.. autoclass:: csbdeep.data.Transform
    :members:

Other transforms
^^^^^^^^^^^^^^^^

.. autofunction:: csbdeep.data.permute_axes


.. ------------

.. .. automodule:: csbdeep.data
..    :members:

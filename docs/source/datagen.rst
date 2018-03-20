Generating training data
========================

In order to create the necessary data for training CARE networks, we
first need to specify matching pairs of raw source and target images.
To that end, we provide the function :func:`csbdeep.datagen.get_tiff_pairs_from_folders`,
which expects TIFF files with the same name, but in different folders.

.. note::
    - The raw data should be representative of all images that the CARE network
      will potentially be applied to after training.
    - Source and target images must be well-aligned to obtain effective CARE networks.

With the raw data specified as above, the function :func:`csbdeep.datagen.create_patches`
can be used to randomly extract patches of a given size that are suitable for training.
By default, patches are normalized based on a range of percentiles
computed on the raw images, which tends to lead to more robust CARE networks in our experience.
If not specified otherwise, patches which are purely background are also excluded
from being extracted, since they do not contain interesting structures.

.. todo::
   Training data generation currently only for real/real data.

.. autofunction:: csbdeep.datagen.get_tiff_pairs_from_folders
.. autofunction:: csbdeep.datagen.create_patches
.. todo::
    Is :func:`csbdeep.datagen.create_patches` a good name?

.. autofunction:: csbdeep.datagen.no_background_patches
.. autofunction:: csbdeep.datagen.sample_percentiles

Advanced topics
---------------

Custom data loaders
###################

If you cannot or do not want to put your raw images in a folder structure
as required by :func:`csbdeep.datagen.get_tiff_pairs_from_folders`, you
can create your custom :obj:`csbdeep.datagen.RawData` object to return
corresponding raw images in the required format for :func:`csbdeep.datagen.create_patches`.

.. autoclass:: csbdeep.datagen.RawData
    :members:

Data augmention
###############

Instead of recording raw images where structures of interest appear in all
possible appearance variations, it can be easier to augment the raw dataset
by included some of those variations that can be easily synthesized. A typical example are
axis-aligned rotations if structures of interest can appear at arbitrary rotations.
To that end, a :obj:`csbdeep.datagen.Transform` object can be used to specify such transformations
to augment the raw dataset.
We currently haven't implemented any transformations, but plan to at least
add axis-aligned rotations and flips later.

.. autoclass:: csbdeep.datagen.Transform
    :members:


.. ------------

.. .. automodule:: csbdeep.datagen
..    :members:
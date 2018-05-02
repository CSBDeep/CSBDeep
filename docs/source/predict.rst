Applying CARE networks
======================

After training a CARE network (see :doc:`training`), we can load and
apply it to raw images that we want to restore.
(Note that this is an alternative to exporting the model and using it with our
`Fiji Plugin <https://github.com/CSBDeep/CSBDeep/wiki/Your-Model-in-Fiji>`_.)

We first create a CARE model with the same name that we have previously
trained it. By not providing a configuration (``config = None``), it will
automatically be loaded from the model's folder. The model's parameters (found
via training) are loaded by invoking :func:`csbdeep.models.CARE.load_weights`.

Before the model can be applied to a raw input image, we first need to specify
image normalization_ and resizing_ methods. In most cases, you want to use
:class:`csbdeep.predict.PercentileNormalizer` with sensible low and high
percentile values (these should be compatible to those used for :doc:`training
data generation <datagen>`).
Furthermore, although a typical CARE model can be applied to various image
sizes after training, some image dimensions must be divisible by powers of
two, depending on the *depth* of the neural network. To that end, we recommend
to use :class:`csbdeep.predict.PadAndCropResizer`, which, if necessary, will
enlarge the image by a few pixels before prediction and remove those
additional pixels afterwards, such that the size of the raw input image is
retained.



**Example**

>>> from tifffile import imread
>>> from csbdeep.models import CARE
>>> from csbdeep.predict import PercentileNormalizer, PadAndCropResizer
>>> model = CARE(config=None, name='my_model')
>>> model.load_weights()
>>> x = imread('my_image.tif')
>>> x_axes = 'YX'
>>> normalizer = PercentileNormalizer(3,99.8)
>>> resizer = PadAndCropResizer()
>>> restored = model.predict(x, x_axes, normalizer, resizer)

.. .. automethod:: csbdeep.models.CARE.predict
.. autoclass:: csbdeep.models.CARE
    :members: predict, predict_probabilistic, load_weights

.. .. autofunction:: csbdeep.predict.tiled_prediction


.. autoclass:: csbdeep.models.IsotropicCARE
    :members: predict, predict_probabilistic


Normalization
-------------

All normalization methods must subclass :class:`csbdeep.predict.Normalizer`.

.. autoclass:: csbdeep.predict.Normalizer
    :members:

.. autoclass:: csbdeep.predict.NoNormalizer

.. autoclass:: csbdeep.predict.PercentileNormalizer
    :members:


Resizing
--------

All resizing methods must subclass :class:`csbdeep.predict.Resizer`.

.. autoclass:: csbdeep.predict.Resizer
    :members:

.. autoclass:: csbdeep.predict.NoResizer

.. autoclass:: csbdeep.predict.PadAndCropResizer
    :members:



Training CARE networks
======================

.. .. image:: https://i.imgflip.com/275svf.jpg

Given suitable training data (see :doc:`datagen`), we can define
and train a CARE network (referred to as *model*) to restore the source data.
To that end, we first need to specify all the options of the model
by creating a configuration object via :class:`csbdeep.models.Config`.
Note that we provide sensible default configuration options that should
work in many cases. However, you can overwrite them via
`keyword arguments <https://docs.python.org/3/glossary.html#term-argument>`_.

The CARE model is instantiated via :class:`csbdeep.models.CARE`
and can be trained with the :func:`csbdeep.models.CARE.train` method.
After training, the learned model can be exported via
:func:`csbdeep.models.CARE.export_TF` to be used with our
`Fiji Plugin <https://github.com/CSBDeep/CSBDeep/wiki/Your-Model-in-Fiji>`_.

**Example**

>>> from csbdeep.tf import limit_gpu_memory
>>> from csbdeep.train import load_data
>>> from csbdeep.models import Config, CARE
>>> limit_gpu_memory(fraction=0.75)
>>> (X,Y), (X_val,Y_val), axes = load_data('my_data.npz', validation_split=0.1)
>>> config = Config(axes, probabilistic=True, unet_n_depth=3)
>>> model = CARE(config, 'my_model')
>>> model.train(X,Y, validation_data=(X_val,Y_val))
>>> model.export_TF()

.. autoclass:: csbdeep.models.Config
    :members:
.. autoclass:: csbdeep.models.CARE
    :members: export_TF, prepare_for_training, train

.. autofunction:: csbdeep.train.load_data
.. autofunction:: csbdeep.nets.common_unet
.. autofunction:: csbdeep.train.prepare_model
.. autofunction:: csbdeep.tf.export_SavedModel
.. autofunction:: csbdeep.tf.limit_gpu_memory
.. .. autofunction:: csbdeep.tf.CARETensorBoard

.. ------
.. .. automodule:: csbdeep.models
..     :members:
.. .. automodule:: csbdeep.nets
..    :members:
.. .. automodule:: csbdeep.blocks
..    :members:
.. .. automodule:: csbdeep.train
..    :members:
.. .. automodule:: csbdeep.losses
..    :members:
.. ------


.. Advanced topics
.. ---------------
..
.. .. todo::
..     - ``ReLU`` as last activation → use :func:`csbdeep.train.prepare_model` with (``loss_bg_thresh``, ``loss_bg_thresh``, ``Y``) and :class:`.. csbdeep.train.ParameterDecayCallback` callback...
..
..
.. .. note::
..     In principle, we can support other backends than TensorFlow for training, but currently not implemented.
..     Futhermore, we use some TF-specific functions, which are in ``csbdeep.tf``.


Isotropic reconstruction
------------------------

Training a CARE network for isotropic reconstruction
(:class:`csbdeep.models.IsotropicCARE`) does not differ from that of a
standard CARE model. What changes is the way in which the training data is
generated (see :doc:`datagen`).

.. autoclass:: csbdeep.models.IsotropicCARE
    :members:

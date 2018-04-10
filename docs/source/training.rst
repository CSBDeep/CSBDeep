Training CARE networks
======================

.. image:: https://i.imgflip.com/275svf.jpg

Defining a network
------------------

Note that a CARE network is referred to as `model` here.

.. autofunction:: csbdeep.nets.common_model
.. autofunction:: csbdeep.nets.common_model_by_name

.. If not enough, use :func:`csbdeep.nets.net_model` or build your own.

.. .. autofunction:: csbdeep.nets.net_model

.. .. automodule:: csbdeep.nets
..    :members:
.. .. automodule:: csbdeep.blocks
..    :members:


Training a network
------------------

See :doc:`/datagen` for data generation.


Preparations
++++++++++++

- Load data
- Optimizer
- Compile
- Callbacks

.. autofunction:: csbdeep.tf.limit_gpu_memory
.. autofunction:: csbdeep.train.load_data
.. autofunction:: csbdeep.train.prepare_model
.. autofunction:: csbdeep.tf.MyTensorBoard


Train
+++++

Takes time...

.. note::
    - ``Probabilistic`` may take a bit longer to train.


Export
++++++
.. autofunction:: csbdeep.tf.export_SavedModel


.. .. automodule:: csbdeep.train
..    :members:
.. .. automodule:: csbdeep.losses
..    :members:



Advanced topics
---------------

.. todo::
    - ``ReLU`` as last activation → use :func:`csbdeep.train.prepare_model` with (``loss_bg_thresh``, ``loss_bg_thresh``, ``Y``) and :class:`csbdeep.train.ParameterDecayCallback` callback.


.. note::
    In principle, we can support other backends than TensorFlow for training, but currently not implemented.
    Futhermore, we use some TF-specific functions, which are in ``csbdeep.tf``.

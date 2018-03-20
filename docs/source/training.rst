Training CARE networks
======================

.. todo::
    Work in progress...

Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod
tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam,
quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo
consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse
cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non
proident, sunt in culpa qui officia deserunt mollit anim id est laborum.


Defining a network
------------------

Model = Network

.. automodule:: csbdeep.nets
   :members:

.. automodule:: csbdeep.blocks
   :members:


Training a network
------------------

.. automodule:: csbdeep.train
   :members:

.. automodule:: csbdeep.losses
   :members:


Backends
--------

In principle, we support other backends than Tensorflow, but this hasn't been tested.
Futhermore, there are some TF-specific functions that we use.

.. automodule:: csbdeep.tf
   :members:

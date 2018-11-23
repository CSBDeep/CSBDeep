Installation
============

Please first `install TensorFlow <https://www.tensorflow.org/install/>`_
by following the official instructions.

.. Note::
   We strongly recommend to use TensorFlow on a Linux or Windows system with a modern
   `CUDA-enabled GPU <https://en.wikipedia.org/wiki/CUDA#GPUs_supported>`_ from Nvidia,
   since TensorFlow currently does not support GPUs on the Mac.
   Without a GPU, training and prediction will be *much* slower (~30-60 times, even when using a computer with 40 CPU cores).

Second, we suggest to `install Jupyter <http://jupyter.org/install>`_ to be able to
run our provided example notebooks that contain step-by-step instructions on how to use this package.

Finally, install the latest stable version of the CSBDeep package with **pip**: ::

    pip install csbdeep


.. Note::
    - The package is compatible with Python 2 and 3, but mainly developed and tested with Python 3 (which we recommend to use).
    - If you use Python 3, you may need to use ``pip3`` instead of ``pip``.


.. .. Note::
..     If you always want the latest version (which might be unstable),
..     you can clone the repository and install it locally: ::

..         git clone https://github.com/csbdeep/csbdeep.git
..         pip install -e csbdeep


.. Todo::
    - Enable installation via **conda** and `conda forge <https://conda-forge.org/>`_.
    - Provide `docker container <https://www.docker.com/what-docker>`_ to avoid potential installation issues.

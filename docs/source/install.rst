Installation
============

Please first `install TensorFlow <https://www.tensorflow.org/install/>`_
by following the official instructions.
If at all possible, we strongly recommend to use TensorFlow on a Linux system with a modern GPU from Nvidia.
Without a GPU, everything will be *much* slower (e.g. 30-60 times, even when using 40 CPU cores).

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

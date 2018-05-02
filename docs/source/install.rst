Installation
============

Please first install TensorFlow by following the
`official instructions <https://www.tensorflow.org/install/>`_.
If possible, we strongly recommend to use TensorFlow with GPU support.

Second, we suggest to `install Jupyter <http://jupyter.org/install>`_ to be able to
run our provided notebooks that contain step-by-step instructions on how to use this package.

Finally, assuming that you have been given access to the CSBDeep_code_ repository,
you can install our Python package by first cloning the repository and
then installing it locally: ::

    git clone https://github.com/mpicbg-csbd/CSBDeep_code.git
    pip install -e CSBDeep_code

.. Note::
    - The code should be compatible with Python 2 and 3, but is mainly developed and tested with Python 3 (which we recommend to use).
    - If you do not use the Anaconda_ Python distribution, you may need to use ``pip3`` instead of ``pip`` if you are using Python 3.
    - You *must not* delete the cloned ``CSBDeep_code`` folder after installation.
      In fact, you can pull from the repository (``git pull``) to update the package.

.. _CSBDeep_code: https://github.com/mpicbg-csbd/CSBDeep_code
.. _Anaconda: https://www.anaconda.com/distribution/

.. Todo::
    Provide `docker container <https://www.docker.com/what-docker>`_ to avoid potential installation issues.
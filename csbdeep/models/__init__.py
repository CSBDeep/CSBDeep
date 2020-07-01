from __future__ import absolute_import, print_function

# checks
try:
    import tensorflow
    del tensorflow
except ModuleNotFoundError as e:
    from six import raise_from
    raise_from(RuntimeError('Please install TensorFlow: https://www.tensorflow.org/install/'), e)


import tensorflow, keras, sys
from distutils.version import LooseVersion
_tf_version = LooseVersion(tensorflow.__version__)
_keras_version = LooseVersion(keras.__version__)
# print(_tf_version)
if  _tf_version >= LooseVersion("2.0.0"):
    print("Found TensorFlow 2 (version %s), which is not fully supported by CSBDeep at the moment (no TensorBoard and model export)." % _tf_version, file=sys.stderr)
    if _keras_version < LooseVersion("2.3.0"):
        raise ImportError("Found Keras version %s, but only versions >= 2.3.0 support TensorFlow 2." % _keras_version)
del tensorflow, keras, LooseVersion, sys



try:
    import keras
    del keras
except ModuleNotFoundError as e:
    if e.name in {'theano','cntk'}:
        from six import raise_from
        raise_from(RuntimeError(
            "Keras is configured to use the '%s' backend, which is not installed. "
            "Please change it to use 'tensorflow' instead: "
            "https://keras.io/getting-started/faq/#where-is-the-keras-configuration-file-stored" % e.name
        ), e)
    else:
        raise e

import keras.backend as K
if K.backend() != 'tensorflow':
    raise NotImplementedError(
            "Keras is configured to use the '%s' backend, which is currently not supported. "
            "Please configure Keras to use 'tensorflow' instead: "
            "https://keras.io/getting-started/faq/#where-is-the-keras-configuration-file-stored" % K.backend()
        )
if K.image_data_format() != 'channels_last':
    raise NotImplementedError(
        "Keras is configured to use the '%s' image data format, which is currently not supported. "
        "Please change it to use 'channels_last' instead: "
        "https://keras.io/getting-started/faq/#where-is-the-keras-configuration-file-stored" % K.image_data_format()
    )
del K


# imports
from .config import BaseConfig, Config
from .base_model import BaseModel
from .care_standard import CARE
from .care_upsampling import UpsamplingCARE
from .care_isotropic import IsotropicCARE
from .care_projection import ProjectionConfig, ProjectionCARE

from __future__ import absolute_import, print_function

# checks
try:
    import tensorflow
    del tensorflow
except ModuleNotFoundError as e:
    from six import raise_from
    raise_from(RuntimeError('Please install TensorFlow: https://www.tensorflow.org/install/'), e)


import tensorflow, sys
from ..utils.tf import keras_import, IS_TF_1

if IS_TF_1:
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

    K = keras_import('backend')
    if K.backend() != 'tensorflow':
        raise NotImplementedError(
                "Keras is configured to use the '%s' backend, which is currently not supported. "
                "Please configure Keras to use 'tensorflow' instead: "
                "https://keras.io/getting-started/faq/#where-is-the-keras-configuration-file-stored" % K.backend()
            )
# else:
#     print("Found TensorFlow 2 (version %s), which might cause issues with CSBDeep." % tensorflow.__version__, file=sys.stderr)

K = keras_import('backend')
if K.image_data_format() != 'channels_last':
    raise NotImplementedError(
        "Keras is configured to use the '%s' image data format, which is currently not supported. "
        "Please change it to use 'channels_last' instead: "
        "https://keras.io/getting-started/faq/#where-is-the-keras-configuration-file-stored" % K.image_data_format()
    )
del tensorflow, sys, keras_import, IS_TF_1, K


# imports
from .config import BaseConfig, Config
from .base_model import BaseModel
from .care_standard import CARE
from .care_upsampling import UpsamplingCARE
from .care_isotropic import IsotropicCARE
from .care_projection import ProjectionConfig, ProjectionCARE
from .pretrained import register_model, register_aliases, clear_models_and_aliases

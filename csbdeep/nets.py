from __future__ import print_function, unicode_literals, absolute_import, division

from keras.layers import Input, Conv2D, Conv3D, Activation
from keras.models import Model
from keras.layers.merge import Add, Concatenate
import keras.backend as K
from .blocks import unet_block
import re


def net_model(input_shape,
              last_activation,
              n_depth=2,
              n_filter_base=16,
              kernel_size=(3,3,3),
              n_conv_per_depth=2,
              activation="relu",
              batch_norm=False,
              dropout=0.0,
              pool_size=(2,2,2),
              n_channel_out=1,
              residual=False,
              prob_out=False):

    if last_activation is None:
        raise ValueError("last activation has to be given (e.g. 'sigmoid', 'relu')!")

    if K.image_data_format() == "channels_last":
        channel_axis = -1
    else:
        channel_axis = 1

    n_dim = len(kernel_size)
    conv = Conv2D if n_dim==2 else Conv3D

    input = Input(input_shape, name = "input")
    unet = unet_block(n_depth, n_filter_base, kernel_size,
                      activation=activation, dropout=dropout, batch_norm=batch_norm,
                      n_conv_per_depth=n_conv_per_depth, pool=pool_size)(input)

    final = conv(n_channel_out, (1,)*n_dim, activation='linear')(unet)
    if residual:
        if not (n_channel_out == input_shape[-1] if K.image_data_format() == "channels_last" else n_channel_out == input_shape[0]):
            raise ValueError("number of input and output channels must be the same for a residual net.")
        final = Add()([final, input])
    final = Activation(activation=last_activation)(final)

    if prob_out:
        scale = conv(n_channel_out, (1,)*n_dim, activation='softplus')(unet)
        final = Concatenate(axis=channel_axis)([final,scale])

    return Model(inputs=input, outputs=final)



def common_model(n_dim=2, n_depth=1, kern_size=3, n_first=16, n_channel_out=1, residual=True, prob_out=False):
    """
    construct a common CARE neural net based on U-Net [1] to be used for image restoration/enhancement

    Parameters
    ----------
    n_dim: int
        number of image dimensions (2 or 3)
    n_depth: int
        number of resolution levels of U-Net architecture
    kern_size: int
        size of convolution filter in all image dimensions
    n_first: int
        number of convolution filters for first U-Net resolution level (value is doubled after each downsampling operation)
    n_channel_out: int
        number of channels of the predicted output image
    residual: bool
        if True, model will internally predict the residual w.r.t. the input (typically better)
        requires number of input and output image channels to be equal
    prob_out: bool
        standard regression (False) or probabilistic prediction (True)
        if True, model will predict two values for each input pixel (mean and positive scale value)

    Returns
    -------
    function to construct the network, which takes as argument the shape of the input image

    Example
    -------
    >>> model = common_model(2, 1,3,16, 1, True, False)(input_shape)

    References
    ----------
    [1] Olaf Ronneberger, Philipp Fischer, Thomas Brox, *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015
    """
    def _build_this(input_shape):
        return net_model(input_shape, "relu", n_depth, n_first, (kern_size,)*n_dim, pool_size=(2,)*n_dim, n_channel_out=n_channel_out, residual=residual, prob_out=prob_out)
    return _build_this



modelname = re.compile("^(?P<model>resunet|unet)(?P<n_dim>\d)(?P<prob_out>p)?_(?P<n_depth>\d+)_(?P<kern_size>\d+)_(?P<n_first>\d+)(_(?P<n_channel_out>\d+)out)?$")
def common_model_by_name(model):
    """
    shorthand notation to call `common_model`

    Parameters
    ----------
    model: string
        define model to be created via string, which is parsed as a regular expression:
        ^(?P<model>resunet|unet)(?P<n_dim>\d)(?P<prob_out>p)?_(?P<n_depth>\d+)_(?P<kern_size>\d+)_(?P<n_first>\d+)(_(?P<n_channel_out>\d+)out)?$

    Returns
    -------
    call to `common_model`

    Example
    -------
    >>> model = common_model_by_name('resunet2_1_3_16_1out')(input_shape)
    >>> # equivalent to: model = common_model(2, 1,3,16, 1, True, False)(input_shape)
    """
    m = modelname.fullmatch(model)
    if m is None:
        raise ValueError("model '%s' unknown, must follow pattern '%s'" % (model, modelname.pattern))
    options = {k:int(m.group(k)) for k in ['n_depth','n_first','kern_size']}
    options['prob_out'] = m.group('prob_out') is not None
    options['residual'] = {'unet': False, 'resunet': True}[m.group('model')]
    options['n_dim'] = int(m.group('n_dim'))
    options['n_channel_out'] = 1 if m.group('n_channel_out') is None else int(m.group('n_channel_out'))

    return common_model(**options)
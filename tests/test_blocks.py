import numpy as np
from csbdeep.utils.tf import keras_import
K = keras_import('backend')
Input = keras_import('layers', 'Input')
Model = keras_import('models', 'Model')

from csbdeep.internals.blocks import unet_block, resnet_block, fpn_block


def test_blocks():
    for ndim in (2,3):
        shape = (64,)*ndim+(1,)

        x = np.zeros((1,)+shape)

        inp = Input(shape)

        block_kwargs = dict(kernel_size=(3,)*ndim, pool=(2,)*ndim)

        for block in (unet_block, fpn_block):
            out = block(**block_kwargs)(inp)
            model = Model(inp, out)
            y = model.predict(x)
            print(block, y.shape)



if __name__ == '__main__':

    test_blocks()



import numpy as np
from csbdeep.utils.tf import keras_import
K = keras_import('backend')
Input = keras_import('layers', 'Input')
Model = keras_import('models', 'Model')
from csbdeep.internals.blocks import unet_block, resnet_block, fpn_block


def compute_receptive_field(model, ndim):
    # TODO: good enough?
    img_size = (128,)*ndim
    # print(img_size)
    mid = tuple(s//2 for s in img_size)
    x = np.zeros((1,)+img_size+(1,), dtype=np.float32)
    z = np.zeros_like(x)
    x[(0,)+mid+(slice(None),)] = 1
    y  = model.predict(x)[0][...,0]
    y0 = model.predict(z)[0][...,0]
    ind = np.where(np.abs(y-y0)>0)
    return [(m-np.min(i), np.max(i)-m) for (m,i) in zip(mid,ind)]

    

def test_blocks():
    for ndim in (2,3):
        shape = (64,)*ndim+(1,)

        x = np.zeros((1,)+shape)

        inp = Input((None,)*ndim+(1,))

        block_kwargs = dict(n_depth=3, kernel_size=(3,)*ndim, pool=(2,)*ndim)

        for block in (unet_block, fpn_block):
            print(f"{block} and {ndim} dims")
            out = block(**block_kwargs)(inp)
            model = Model(inp, out)
            print(f"receptive field: {compute_receptive_field(model, ndim)}")
            y = model.predict(x)



if __name__ == '__main__':

    model = test_blocks()



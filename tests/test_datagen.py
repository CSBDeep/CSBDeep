from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

# import warnings
import numpy as np

from tqdm import tqdm
import pytest


def test_create_patches():
    from csbdeep.data import RawData, create_patches
    rng = np.random.RandomState(42)
    def get_data(n_images, axes, shape):
        def _gen():
            for i in range(n_images):
                x = rng.uniform(size=shape)
                y = 5 + 3*x
                yield x, y, axes, None
        return RawData(_gen, n_images, '')

    n_images, n_patches_per_image = 2, 4
    def _create(img_size,img_axes,patch_size,patch_axes):
        X,Y,XYaxes = create_patches (
            raw_data            = get_data(n_images, img_axes, img_size),
            patch_size          = patch_size,
            patch_axes          = patch_axes,
            n_patches_per_image = n_patches_per_image,
        )
        assert len(X) == n_images*n_patches_per_image
        assert np.allclose(X,Y,atol=1e-6)
        if patch_axes is not None:
            assert XYaxes == 'SC'+patch_axes.replace('C','')

    _create((128,128),'YX',(32,32),'YX')
    _create((128,128),'YX',(32,32),None)
    _create((128,128),'YX',(32,32),'XY')
    _create((128,128),'YX',(32,32,1),'XYC')

    _create((32,48,32),'ZYX',(16,32,8),None)
    _create((32,48,32),'ZYX',(16,32,8),'ZYX')
    _create((32,48,32),'ZYX',(16,32,8),'YXZ')
    _create((32,48,32),'ZYX',(16,32,1,8),'YXCZ')


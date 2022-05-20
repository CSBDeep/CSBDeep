from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

# import warnings
import numpy as np
import pytest
from tifffile import imread
try:
    from tifffile import imwrite as imsave
except ImportError:
    from tifffile import imsave
from csbdeep.data import RawData, create_patches, create_patches_reduced_target
from csbdeep.io import load_training_data
from csbdeep.utils import Path, axes_dict, move_image_axes, backend_channels_last



def test_create_patches():
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



def test_create_patches_reduced_target():
    rng = np.random.RandomState(42)
    def get_data(n_images, axes, shape):
        red_n = rng.choice(len(axes)-1)+1
        red_axes = ''.join(rng.choice(tuple(axes),red_n,replace=False))
        keepdims = rng.choice((True,False))

        def _gen():
            for i in range(n_images):
                x = rng.uniform(size=shape)
                y = np.mean(x,axis=tuple(axes_dict(axes)[a] for a in red_axes),keepdims=keepdims)
                yield x, y, axes, None
        return RawData(_gen, n_images, ''), red_axes, keepdims

    n_images, n_patches_per_image = 2, 4
    def _create(red_none,img_size,img_axes,patch_size,patch_axes):
        raw_data, red_axes, keepdims = get_data(n_images, img_axes, img_size)
        # change patch_size to (img_size or None) for red_axes
        patch_size = list(patch_size)
        for a in red_axes:
            patch_size[axes_dict(img_axes if patch_axes is None else patch_axes)[a]] = (
                None if red_none else img_size[axes_dict(img_axes)[a]]
            )
        X,Y,XYaxes = create_patches_reduced_target (
            raw_data            = raw_data,
            patch_size          = patch_size,
            patch_axes          = patch_axes,
            n_patches_per_image = n_patches_per_image,
            reduction_axes      = red_axes,
            target_axes         = rng.choice((None,img_axes)) if keepdims else ''.join(a for a in img_axes if a not in red_axes),
            #
            normalization       = lambda patches_x, patches_y, *args: (patches_x, patches_y),
            verbose             = False,
        )
        assert len(X) == n_images*n_patches_per_image
        _X = np.mean(X,axis=tuple(axes_dict(XYaxes)[a] for a in red_axes),keepdims=True)
        err = np.max(np.abs(_X-Y))
        assert err < 1e-5

    for b in (True,False):
        _create(b,(128,128),'YX',(32,32),'YX')
        _create(b,(128,128),'YX',(32,32),None)
        _create(b,(128,128),'YX',(32,32),'XY')
        _create(b,(128,128),'YX',(32,32,1),'XYC')

        _create(b,(32,48,32),'ZYX',(16,32,8),None)
        _create(b,(32,48,32),'ZYX',(16,32,8),'ZYX')
        _create(b,(32,48,32),'ZYX',(16,32,8),'YXZ')
        _create(b,(32,48,32),'ZYX',(16,32,1,8),'YXCZ')

        _create(b,(128,2,128),'YCX',(32,2,32),'YCX')
        _create(b,(3,128,128),'CYX',(3,32,32),None)
        _create(b,(128,128,4),'YXC',(4,32,32),'CXY')
        _create(b,(128,128,5),'YXC',(32,32,5),'XYC')

        _create(b,(32,48,2,32),'ZYCX',(16,32,2,8),None)
        _create(b,(32,3,48,32),'ZCYX',(3,16,32,8),'CZYX')
        _create(b,(4,32,48,32),'CZYX',(16,32,8,4),'YXZC')
        _create(b,(32,48,32,2),'ZYXC',(16,32,2,8),'YXCZ')



def test_create_save_and_load(tmpdir):
    rng = np.random.RandomState(42)
    tmpdir = Path(str(tmpdir))
    save_file = str(tmpdir / 'data.npz')

    n_images, n_patches_per_image = 2, 4
    def _create(img_size,img_axes,patch_size,patch_axes):
        U,V = (rng.uniform(size=(n_images,)+img_size) for _ in range(2))
        X,Y,XYaxes = create_patches (
            raw_data            = RawData.from_arrays(U,V,img_axes),
            patch_size          = patch_size,
            patch_axes          = patch_axes,
            n_patches_per_image = n_patches_per_image,
            save_file           = save_file
        )
        (_X,_Y), val_data, _XYaxes = load_training_data(save_file,verbose=True)
        assert val_data is None
        assert _XYaxes[-1 if backend_channels_last else 1] == 'C'
        _X,_Y = (move_image_axes(u,fr=_XYaxes,to=XYaxes) for u in (_X,_Y))
        assert np.allclose(X,_X,atol=1e-6)
        assert np.allclose(Y,_Y,atol=1e-6)
        assert set(XYaxes) == set(_XYaxes)
        assert load_training_data(save_file,validation_split=0.5)[2] is not None
        assert all(len(x)==3 for x in load_training_data(save_file,n_images=3)[0])

    _create((  64,64), 'YX',(16,16  ),None)
    _create((  64,64), 'YX',(16,16  ),'YX')
    _create((  64,64), 'YX',(16,16,1),'YXC')
    _create((1,64,64),'CYX',(  16,16),'YX')
    _create((1,64,64),'CYX',(1,16,16),None)
    _create((64,3,64),'YCX',(3,16,16),'CYX')
    _create((64,3,64),'YCX',(16,16,3),'YXC')



def test_rawdata_from_folder(tmpdir):
    rng = np.random.RandomState(42)
    tmpdir = Path(str(tmpdir))

    n_images, img_size, img_axes = 3, (64,64), 'YX'
    data = {'X' : rng.uniform(size=(n_images,)+img_size).astype(np.float32),
            'Y' : rng.uniform(size=(n_images,)+img_size).astype(np.float32)}

    for name,images in data.items():
        (tmpdir/name).mkdir(exist_ok=True)
        for i,img in enumerate(images):
            imsave(str(tmpdir/name/('img_%02d.tif'%i)),img)

    raw_data = RawData.from_folder(str(tmpdir),['X'],'Y',img_axes)
    assert raw_data.size == n_images
    for i,(x,y,axes,mask) in enumerate(raw_data.generator()):
        assert mask is None
        assert axes == img_axes
        assert any(np.allclose(x,u) for u in data['X'])
        assert any(np.allclose(y,u) for u in data['Y'])

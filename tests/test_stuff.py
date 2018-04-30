from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

# import warnings
import numpy as np
from itertools import product
import tempfile

from csbdeep.utils import axes_dict
from csbdeep.models import Config, CARE
from csbdeep.predict import NoNormalizer, NoResizer, tile_overlap
from csbdeep.nets import receptive_field_unet

from keras import backend as K
from tqdm import tqdm
import pytest

def config_generator(**kwargs):
    assert 'axes' in kwargs
    keys, values = kwargs.keys(), kwargs.values()
    values = [v if isinstance(v,(list,tuple)) else [v] for v in values]
    for p in product(*values):
        yield Config(**dict(zip(keys,p)))


def test_config():
    assert K.image_data_format() in ('channels_first','channels_last')
    def _with_channel(axes):
        axes = axes.upper()
        if 'C' in axes:
            return axes
        return (axes+'C') if K.image_data_format() == 'channels_last' else ('C'+axes)

    axes_list = [
        ('yx',_with_channel('YX')),
        ('YX',_with_channel('YX')),
        ('XYZ',_with_channel('XYZ')),
        ('SYX',_with_channel('YX')),
        ('SXYZ',_with_channel('XYZ')),
        (_with_channel('YX'),_with_channel('YX')),
        (_with_channel('XYZ'),_with_channel('XYZ')),
        (_with_channel('SYX'),_with_channel('YX')),
        (_with_channel('SXYZ'),_with_channel('XYZ')),
    ]

    for (axes,axes_ref) in axes_list:
        assert Config(axes).axes == axes_ref

    with pytest.raises(ValueError):
        Config('XYC')
        Config('CXY')
    with pytest.raises(ValueError):
        Config('XYZC')
        Config('CXYZ')
    with pytest.raises(ValueError): Config('XTY')
    with pytest.raises(ValueError): Config('XYZT')
    with pytest.raises(ValueError): Config('XYS')
    with pytest.raises(ValueError): Config('XSYZ')


def test_model_build():
    configs = config_generator(
        axes                  = ['YX','ZYX'],
        n_channel_in          = [1,2],
        n_channel_out         = [1,2],
        probabilistic         = [False,True],
        unet_residual         = [False,True],
        unet_n_depth          = [1,2],
        # unet_kern_size        = [3],
        # unet_n_first          = [32],
        # unet_last_activation  = ['linear'],
        # unet_input_shape      = [(None, None, 1)],
        #
        # train_batch_size      = [16],
        # train_checkpoint      = ['weights_best.h5'],
        # train_epochs          = [100],
        # train_learning_rate   = [0.0004],
        # train_loss            = ['mae'],
        # train_reduce_lr       = [{'factor': 0.5, 'patience': 10}],
        # train_steps_per_epoch = [400],
        # train_tensorboard     = [True],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        for config in configs:
            K.clear_session()
            def _build():
                CARE(config,outdir=tmpdir)
            if config.is_valid():
                _build()
            else:
                with pytest.raises(ValueError):
                    _build()


def test_model_train():
    rng = np.random.RandomState(42)
    configs = config_generator(
        axes                  = ['YX','ZYX'],
        n_channel_in          = [1,2],
        n_channel_out         = [1,2],
        probabilistic         = [False,True],
        # unet_residual         = [False,True],
        unet_n_depth          = [1],

        unet_kern_size        = [3],
        unet_n_first          = [4],
        unet_last_activation  = ['linear'],
        # unet_input_shape      = [(None, None, 1)],

        train_loss            = ['mae','laplace'],
        train_epochs          = [2],
        train_steps_per_epoch = [2],
        # train_learning_rate   = [0.0004],
        train_batch_size      = [2],
        # train_tensorboard     = [True],
        # train_checkpoint      = ['weights_best.h5'],
        # train_reduce_lr       = [{'factor': 0.5, 'patience': 10}],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        for config in configs:
            K.clear_session()
            if config.is_valid():
                X = rng.uniform(size=(4,)+(32,)*config.n_dim+(config.n_channel_in,))
                Y = rng.uniform(size=(4,)+(32,)*config.n_dim+(config.n_channel_out,))
                model = CARE(config,outdir=tmpdir)
                model.train(X,Y,(X,Y))


def test_model_predict():
    rng = np.random.RandomState(42)
    configs = config_generator(
        axes                  = ['YX','ZYX'],
        n_channel_in          = [1,2],
        n_channel_out         = [1,2],
        probabilistic         = [False,True],
        # unet_residual         = [False,True],
        unet_n_depth          = [2],

        unet_kern_size        = [3],
        unet_n_first          = [4],
        unet_last_activation  = ['linear'],
        # unet_input_shape      = [(None, None, 1)],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        normalizer, resizer = NoNormalizer(), NoResizer()

        for config in filter(lambda c: c.is_valid(), configs):
            K.clear_session()
            model = CARE(config,outdir=tmpdir)
            axes = config.axes

            def _predict(imdims,axes):
                img = rng.uniform(size=imdims)
                # print(img.shape, axes, config.n_channel_out)
                mean, scale = model._predict_mean_and_scale(img, axes, normalizer, resizer)
                if config.probabilistic:
                    assert mean.shape == scale.shape
                else:
                    assert scale is None

                if 'C' not in axes:
                    if config.n_channel_out == 1:
                        assert mean.shape == img.shape
                    else:
                        assert mean.shape == img.shape + (config.n_channel_out,)
                else:
                    channel = axes_dict(axes)['C']
                    imdims[channel] = config.n_channel_out
                    assert mean.shape == tuple(imdims)


            imdims = list(rng.randint(20,40,size=config.n_dim))
            div_n = 2**config.unet_n_depth
            imdims = [(d//div_n)*div_n for d in imdims]

            if config.n_channel_in == 1:
                _predict(imdims,axes=axes.replace('C',''))

            channel = rng.randint(0,config.n_dim)
            imdims.insert(channel,config.n_channel_in)
            _axes = axes.replace('C','')
            _axes = _axes[:channel]+'C'+_axes[channel:]
            _predict(imdims,axes=_axes)


def test_model_predict_tiled():
    """
    Test that tiled prediction yields the same
    or similar result as compared to predicting
    the whole image at once.
    """
    rng = np.random.RandomState(42)
    configs = config_generator(
        axes                  = ['YX','ZYX'],
        n_channel_in          = [1],
        n_channel_out         = [1],
        probabilistic         = [False],
        # unet_residual         = [False,True],
        unet_n_depth          = [1,2,3],
        unet_kern_size        = [3,5],

        unet_n_first          = [4],
        unet_last_activation  = ['linear'],
        # unet_input_shape      = [(None, None, 1)],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        normalizer, resizer = NoNormalizer(), NoResizer()

        for config in filter(lambda c: c.is_valid(), configs):
            K.clear_session()
            model = CARE(config,outdir=tmpdir)

            def _predict(imdims,axes,n_tiles):
                img = rng.uniform(size=imdims)
                # print(img.shape, axes)
                mean,       scale       = model._predict_mean_and_scale(img, axes, normalizer, resizer, n_tiles=1)
                mean_tiled, scale_tiled = model._predict_mean_and_scale(img, axes, normalizer, resizer, n_tiles=n_tiles)
                assert mean.shape == mean_tiled.shape
                if config.probabilistic:
                    assert scale.shape == scale_tiled.shape
                error_max = np.max(np.abs(mean-mean_tiled))
                # print('n, k, err = {0}, {1}x{1}, {2}'.format(model.config.unet_n_depth, model.config.unet_kern_size, error_max))
                assert error_max < 1e-3
                return mean, mean_tiled

            imdims = list(rng.randint(100,130,size=config.n_dim))
            if config.n_dim == 3:
                imdims[0] = 32 # make one dim small, otherwise test takes too long
            div_n = 2**config.unet_n_depth
            imdims = [(d//div_n)*div_n for d in imdims]

            n_blocks = np.max(imdims) // div_n
            def _predict_wrapped(imdims,axes,n_tiles):
                if 0 < n_tiles <= n_blocks:
                    _predict(imdims,axes,n_tiles=n_tiles)
                else:
                    with pytest.warns(UserWarning):
                        _predict(imdims,axes,n_tiles=n_tiles)

            imdims.insert(0,config.n_channel_in)
            axes = config.axes.replace('C','')
            # return _predict(imdims,'C'+axes,n_tiles=(3,4))

            # tile one dimension
            for n_tiles in (0,2,3,6,n_blocks+1):
                if config.n_channel_in == 1:
                    _predict_wrapped(imdims[1:],axes,n_tiles)
                _predict_wrapped(imdims,'C'+axes,n_tiles)

            # tile two dimensions
            for n_tiles in product((2,4),(3,5)):
                _predict(imdims,'C'+axes,n_tiles)

            # tile three dimensions
            if config.n_dim == 3:
                _predict(imdims,'C'+axes,(2,3,4))


def test_tile_overlap():
    n_depths = (1,2,3,4,5)
    n_kernel = (3,5,7)
    img_size = 1280
    for k in n_kernel:
        for n in n_depths:
            K.clear_session()
            rf_x, rf_y = receptive_field_unet(n,k,2,img_size)
            assert rf_x == rf_y
            rf = rf_x
            assert np.abs(rf[0]-rf[1]) < 10
            assert sum(rf)+1 < img_size
            assert max(rf) == tile_overlap(n,k)
            # print("receptive field of depth %d and kernel size %d: %s"%(n,k,rf));


def test_datagen():
    pass

def test_transforms():
    pass

def test_exceptions():
    """
    test that exceptions are thrown for illegal function arguments.
    """

def test_bad_bugs():
    """
    think about which mistakes have serious consequences
    and test for catching them, especially if they are easily overlooked.
    """

def test_cpu_gpu_equality():
    """
    probably not our job, but do a quick test
    whether CARE on cpu and gpu give roughly the
    same results.
    """

def test_iso_care():
    """
    typical use of isotropic CARE
    """
    # check that created training patches are registered as best as possible
    # from skimage.feature import register_translation
    # shifts = register_translation(u,x_norm_pad)[0]
    # # assert np.all(shifts==0)


def test_image_scaling():
    """
    Don't give same results: gputools.scale / scipy.ndimage.interpolation.zoom
    Problem?
    """

def test_resizer():
    pass

def test_normalizer():
    pass
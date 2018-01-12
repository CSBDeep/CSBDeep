from __future__ import print_function, unicode_literals, absolute_import, division

import os
import numpy as np
from glob import glob
from tifffile import imread
# import argparse
from time import time
# from deeptools.utils.patches import sample_patches_from_multiple_stacks
# from deeptools.utils import shuffle_inplace, normalize
# from gputools.denoise import nlm3
import numexpr

from tqdm import tqdm
# from six.moves import map
# try:
#     from tqdm import tqdm
# except ImportError:
#     tqdm = lambda x: x
# from numbers import Number

def create_patches(p="../data/planaria/",
        output='GT',
        inputs=['condition_2'],
        # inputs=['condition_2','foo'],
        patch_size=(8,32,32),
        thresh_patch=0.4,
        n_samples=2,
        # norm_percentiles=(1,99),
        norm_percentiles=([0,4],[99.4,99.9]),
        # norm_percentiles=(lambda: np.random.uniform(0,4), lambda: np.random.uniform(99.4,99.9)),
        ):

    ##
    assert isinstance(norm_percentiles,(list,tuple)) and 2==len(norm_percentiles)
    if all(( np.isscalar(_) for _ in norm_percentiles )):
        pmin = lambda: norm_percentiles[0]
        pmax = lambda: norm_percentiles[1]
    elif all(( callable(_) for _ in norm_percentiles )):
        pmin = norm_percentiles[0]
        pmax = norm_percentiles[1]
    elif all(( (isinstance(_,(list,tuple)) and 2==len(_)) for _ in norm_percentiles )):
        pmin = lambda: np.random.uniform(*norm_percentiles[0])
        pmax = lambda: np.random.uniform(*norm_percentiles[1])
    else:
        raise ValueError("bad value for 'norm_percentiles'.")

    ##
    image_names = [os.path.split(f)[-1] for f in glob(os.path.join(p,output,'*.tif*'))]


    assert all((os.path.exists(os.path.join(p,i,n)) for i in inputs for n in image_names))
    image_pairs = [(os.path.join(p,i,n),os.path.join(p,output,n)) for i in inputs for n in image_names]

    # print("found image pairs: %s \n\n"%str(image_pairs))

    # image_pairs = image_pairs[:2] #tmp

    X,Y = [],[]

    ##
    for fx,fy in tqdm(image_pairs):

        x,y = imread(fx),imread(fy)
        #x,y = x[:16,:128,:128],y[:16,:128,:128] #tmp
        assert x.shape == y.shape


        thresh = thresh_patch * np.percentile(y,99.8)
        # n_samples = 8

        _Y, _X = sample_patches_from_multiple_stacks((y, x),
                                                     patch_size=patch_size,
                                                     patch_filter_mode="absolute",
                                                     min_max=thresh,
                                                     n_samples=n_samples,
                                                     augment=False)

        if _X is not None:
            pmins = [pmin() for _ in range(n_samples)]
            pmaxs = [pmax() for _ in range(n_samples)]
            perc_x = np.percentile(x,pmins +         pmaxs, keepdims=True).astype(np.float32)
            perc_y = np.percentile(y,n_samples*[0] + pmaxs, keepdims=True).astype(np.float32)
            # perc_y = np.concatenate((
            #     np.repeat(np.min(y,keepdims=True)[np.newaxis],n_samples,axis=0),
            #     np.percentile(y,pmaxs,keepdims=True)
            # )).astype(np.float32)
            perc_x_min, perc_x_max = perc_x[:n_samples], perc_x[n_samples:]
            perc_y_min, perc_y_max = perc_y[:n_samples], perc_y[n_samples:]
            # print(perc_x.shape)
            # print(perc_y.shape)

            # print([_.shape for _ in (perc_y,perc_x,_Y,_X)])
            # print([_.dtype for _ in (perc_y,perc_x,_Y,_X)])
            # print(np.round(pmins,1))
            # print(np.round(pmaxs,1))

            _X = _X.astype(np.float32,copy=False)
            _Y = _Y.astype(np.float32,copy=False)
            eps = np.float32(1e-20)
            try:
                # raise ImportError()
                import numexpr
                _X = numexpr.evaluate("(_X - perc_x_min) / ( perc_x_max - perc_x_min + eps )")
                _Y = numexpr.evaluate("(_Y - perc_y_min) / ( perc_y_max - perc_y_min + eps )")
            except ImportError:
                _X = (_X - perc_x_min) / ( perc_x_max - perc_x_min + eps )
                _Y = (_Y - perc_y_min) / ( perc_y_max - perc_y_min + eps )

            X.append(_X)
            Y.append(_Y)

        del x, y

        # break

    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)

    # print("found %s samples..."%len(X))
    # print("shuffling...")
    shuffle_inplace(X, Y)

    # X = X[:,np.newaxis]
    # Y = Y[:,np.newaxis]


    # print (X.shape, Y.shape)
    # print (X.dtype, Y.dtype)

    return X,Y













def sample_patches_from_multiple_stacks(datas, patch_size, n_samples,
                                        min_max=.0,
                                        augment=False,
                                        patch_filter=None,
                                        patch_filter_mode="relative",
                                        filter_size=None,
                                        verbose=False):
    """
    sample 3d patches of size dshape from data along axis accoridng to patch_filter

    patch_filter is applied to datas[0]

    signature: patch_filter(data, patch_size)
    """

    if augment:
        return np.concatenate([sample_patches_from_multiple_stacks(np.array(augs),
                                                                   patch_size = patch_size,
                                                                   n_samples = n_samples,
                                                                   augment=False,
                                                                   min_max=min_max,
                                                                   patch_filter_mode=patch_filter_mode,
                                                                   patch_filter=patch_filter,
                                                                   filter_size=filter_size)
                               for augs in zip(*[augment_iter(_d) for _d in datas])], axis=1)

    if not np.all([a.shape == datas[0].shape for a in datas]):
        raise ValueError("all input shapes must be the same: %s" % (" / ".join(str(a.shape) for a in datas)))

    if not np.all([s <= d for d, s in zip(datas[0].shape, patch_size)]):
        raise ValueError("data shape %s is smaller than patch_size %s along some dimensions" % (
            str(datas[0].shape), str(patch_size)))

    if patch_filter is None:
        patch_filter = patch_filter_max(min_max, mode=patch_filter_mode)


    if filter_size is None:
        filter_size = patch_size

    assert np.all([n > 0 for n in filter_size])
    assert np.all([n > 0 for n in patch_size])

    filter_mask = patch_filter(datas[0].astype(np.float32), filter_size)

    # get the valid indices

    border_slices = tuple([slice(s // 2, d - s + s // 2 + 1) for s, d in zip(patch_size, datas[0].shape)])
    valid_inds = np.where(filter_mask[border_slices])

    if len(valid_inds[0]) == 0:
        raise ValueError("could not find any thing to sample from...")

    valid_inds = [v + s.start for s, v in zip(border_slices, valid_inds)]

    # sample



    sample_inds = np.random.randint(0, len(valid_inds[0]), n_samples)

    rand_inds = [v[sample_inds] for v in valid_inds]

    res = [np.stack([data[r[0] - patch_size[0] // 2:r[0] + patch_size[0] - patch_size[0] // 2,
                     r[1] - patch_size[1] // 2:r[1] + patch_size[1] - patch_size[1] // 2,
                     r[2] - patch_size[2] // 2:r[2] + patch_size[2] - patch_size[2] // 2,
                     ] for r in zip(*rand_inds)]) for data in datas]

    return res












def patch_filter_max(min_max, mode="relative"):
    # try:
    #     from gputools import max_filter
    # except ImportError:
    #     raise ImportError("could not import 'gputools'!")
    from scipy.ndimage.filters import maximum_filter

    def _filter_rel(data, patch_size):
        return maximum_filter(data.astype(np.float32), patch_size, mode='constant') >= min_max * np.amax(data)

    def _filter_abs(data, patch_size):
        return maximum_filter(data.astype(np.float32), patch_size, mode='constant') >= min_max

    return _filter_rel if mode == "relative" else _filter_abs








def shuffle_inplace(*arrs):
    rng = np.random.RandomState()
    _state = rng.get_state()
    for a in arrs:
        rng.set_state(_state)
        rng.shuffle(a)

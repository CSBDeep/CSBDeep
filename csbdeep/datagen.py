from __future__ import print_function, unicode_literals, absolute_import, division

# import os
import numpy as np
# from glob import glob
from tifffile import imread
# from time import time
# import numexpr

# from tqdm import tqdm
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, *args, **kwargs: x

from six.moves import map
import collections
from functools import reduce

# https://www.scivision.co/python-idiomatic-pathlib-use/
try:
    from pathlib import Path
    Path().expanduser()
except (ImportError,AttributeError):
    from pathlib2 import Path



def _raise(e):
    raise e

# https://docs.python.org/3/library/itertools.html#itertools-recipes
def consume(iterator):
    collections.deque(iterator, maxlen=0)

def compose(*funcs):
    return lambda x: reduce(lambda f,g: g(f), funcs, x)

# def pipeline(*steps):
#     return reduce(lambda f,g: g(f), steps)


def load_image_pairs(p,output,inputs,pattern='*.tif*'):
    p = Path(p)
    image_names = [f.name for f in (p/output).glob(pattern)]
    consume ((
        (p/i/n).exists() or _raise(FileNotFoundError(p/i/n))
        for i in inputs for n in image_names
    ))
    xy_name_pairs = [(p/i/n, p/output/n) for i in inputs for n in image_names]
    n_images = len(xy_name_pairs)

    def _gen():
        for fx, fy in xy_name_pairs:
            x, y = imread(str(fx)), imread(str(fy))
            # x,y = x[:,256:-256,256:-256],y[:,256:-256,256:-256] #tmp
            x.shape == y.shape or _raise(ValueError())
            yield x, y

    return _gen(), n_images


Transform = collections.namedtuple('Transform',('name','generator','size'))

def tf_identity():
    def _gen(inputs):
        for d in inputs:
            yield d
    return Transform('Identity', _gen, 1)

def tf_flip(axis):
    def _gen(inputs):
        for x,y in inputs:
            axis < x.ndim or _raise(ValueError())
            yield x,y
            yield np.flip(x,axis),np.flip(y,axis)
    return Transform('Flip (axis=%d)'%axis, _gen, 2)


def normalize(x, pmin, pmax, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    mi = np.percentile(x,pmin,axis=axis,keepdims=True)
    ma = np.percentile(x,pmax,axis=axis,keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x   = x.astype(dtype,copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x =                   (x - mi) / ( ma - mi + eps )

    if clip:
        x = np.clip(x,0,1)

    return x



def patch_filter_max(predicate=(lambda image,filtered: filtered > 0.4*np.percentile(image,99.9)), filter_size=None):
    from scipy.ndimage.filters import maximum_filter
    def _filter(image, patch_size, dtype=np.float32):
        if dtype is not None:
            image = image.astype(dtype)
        return predicate(image, maximum_filter(image, (patch_size if filter_size is None else filter_size), mode='constant'))
    return _filter



def create_patches (
        p="../data/planaria/",
        output='GT',
        inputs=['condition_2'],
        # inputs=['condition_2','foo'],
        patch_size=(8,32,32),
        # thresh_patch=0.4,
        n_samples=2,
        # transforms=[tf_flip(0),tf_flip(1)],
        transforms=None,
        patch_filter=patch_filter_max(),
        # percentiles=(1,99),
        percentiles=lambda: (np.random.uniform(0,4), np.random.uniform(99.4,99.9)),
        # percentiles_axis=None, # enable?
        shuffle=True,
        verbose=True,
    ):

    ## percentiles
    _ps_ok = lambda ps: isinstance(ps,(list,tuple,np.ndarray)) and len(ps)==2 and all(map(np.isscalar,ps)) and (0<=ps[0]<ps[1]<=100)
    if callable(percentiles):
        _p_example = percentiles()
        _ps_ok(_p_example) or _raise(ValueError(_p_example))
        norm_percentiles = percentiles
    else:
        _ps_ok(percentiles) or _raise(ValueError(percentiles))
        norm_percentiles = lambda: percentiles

    ## images and transforms
    if transforms is None or len(transforms)==0:
        transforms = (tf_identity(),)
    image_pairs, n_raw_images = load_image_pairs(p,output,inputs)
    tf = Transform(*zip(*transforms)) # convert list of Transforms into Transform of lists
    image_pairs = compose(*tf.generator)(image_pairs) # combine all transformations with raw images as input
    n_transforms = np.prod(tf.size)
    n_images = n_raw_images * n_transforms

    ## summary
    if verbose:
        # print('Transforms = ', list(zip(tf.name,tf.size)), flush=True)
        print('='*66)
        print('{p}: output/target="{output}" inputs={inputs}'.format(p=p,inputs=list(inputs),output=output))
        print('='*66)
        print('%5d raw images x %4d transformations   = %5d images' % (n_raw_images,n_transforms,n_images))
        print('%5d images     x %4d patches per image = %5d patches in total' % (n_images,n_samples,n_images*n_samples))
        print('\nTransformations:')
        for t in transforms:
            print('{t.size} x {t.name}'.format(t=t))
        print('='*66)
        print(flush=True)

    ##
    X,Y = [],[]

    for x,y in tqdm(image_pairs,total=n_images):

        _Y,_X = sample_patches_from_multiple_stacks((y,x), patch_size, n_samples, patch_filter)

        pmins, pmaxs = zip(*(norm_percentiles() for _ in range(n_samples)))
        X.append( normalize_mi_ma(_X, np.percentile(x,pmins,keepdims=True), np.percentile(x,pmaxs,keepdims=True)) )
        Y.append( normalize_mi_ma(_Y, np.min(y),                            np.percentile(y,pmaxs,keepdims=True)) )

        del x,y
        # break

    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)

    if shuffle:
        shuffle_inplace(X, Y)

    return X,Y













def sample_patches_from_multiple_stacks(datas, patch_size, n_samples, patch_filter, verbose=False):
    """
    sample 3d patches of size dshape from data along axis accoridng to patch_filter

    patch_filter is applied to datas[0]

    signature: patch_filter(data, patch_size)
    """

    # if augment:
    #     return np.concatenate([sample_patches_from_multiple_stacks(np.array(augs),
    #                                                                patch_size = patch_size,
    #                                                                n_samples = n_samples,
    #                                                                augment=False,
    #                                                                min_max=min_max,
    #                                                                patch_filter_mode=patch_filter_mode,
    #                                                                patch_filter=patch_filter,
    #                                                                filter_size=filter_size)
    #                            for augs in zip(*[augment_iter(_d) for _d in datas])], axis=1)

    len(patch_size)==datas[0].ndim or _raise(ValueError())

    if not all(( a.shape == datas[0].shape for a in datas )):
        raise ValueError("all input shapes must be the same: %s" % (" / ".join(str(a.shape) for a in datas)))

    if not all(( 0 < s <= d for s,d in zip(patch_size,datas[0].shape) )):
        raise ValueError("patch_size %s negative or larger than data shape %s along some dimensions" % (str(patch_size), str(datas[0].shape)))


    filter_mask = patch_filter(datas[0], patch_size)

    # get the valid indices

    border_slices = tuple([slice(s // 2, d - s + s // 2 + 1) for s, d in zip(patch_size, datas[0].shape)])
    valid_inds = np.where(filter_mask[border_slices])

    if len(valid_inds[0]) == 0:
        raise ValueError("'patch_filter' didn't return any region to sample from")

    valid_inds = [v + s.start for s, v in zip(border_slices, valid_inds)]

    # sample
    sample_inds = np.random.choice(len(valid_inds[0]), n_samples, replace=len(valid_inds[0])<n_samples)

    rand_inds = [v[sample_inds] for v in valid_inds]

    res = [np.stack([data[r[0] - patch_size[0] // 2:r[0] + patch_size[0] - patch_size[0] // 2,
                     r[1] - patch_size[1] // 2:r[1] + patch_size[1] - patch_size[1] // 2,
                     r[2] - patch_size[2] // 2:r[2] + patch_size[2] - patch_size[2] // 2,
                     ] for r in zip(*rand_inds)]) for data in datas]

    return res





















def shuffle_inplace(*arrs):
    rng = np.random.RandomState()
    state = rng.get_state()
    for a in arrs:
        rng.set_state(state)
        rng.shuffle(a)

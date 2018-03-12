from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
from tifffile import imread
from six.moves import map
from collections import namedtuple

from tqdm import tqdm
from .utils import Path, normalize_mi_ma, _raise, consume, compose, shuffle_inplace


## Transforms (to be added later)

Transform = namedtuple('Transform',('name','generator','size'))

def tf_identity():
    def _gen(inputs):
        for d in inputs:
            yield d
    return Transform('Identity', _gen, 1)

# def tf_flip(axis):
#     def _gen(inputs):
#         for x,y in inputs:
#             axis < x.ndim or _raise(ValueError())
#             yield x,y
#             yield np.flip(x,axis),np.flip(y,axis)
#     return Transform('Flip (axis=%d)'%axis, _gen, 2)



## Input data

InputData = namedtuple('InputData',('generator','size','description'))

def get_tiff_pairs_from_folders(basepath,input_dirs,output_dir='GT',pattern='*.tif*'):
    """Get pairs of corresponding TIFF images read from folders.

    Two images correspond to each other if they have the same file name, but are located in different folders.

    Parameters
    ----------
    basepath : str
        Base folder that contains sub-folders with images.
    input_dirs : iterable
        List of folder names relative to `basepath` that contain the input/source images (e.g., with low SNR).
    output_dir : str
        Folder name relative to `basepath` that contains the output/target images (e.g., with high SNR).
    pattern : str
        Glob pattern to match the desired TIFF images.

    Returns
    -------
    InputData

    Raises
    ------
    FileNotFoundError
        If an image found in `output_dir` does not exist in all `input_dirs`.
    ValueError
        If corresponding images do not have the same size.

    """
    p = Path(basepath)
    image_names = [f.name for f in (p/output_dir).glob(pattern)]
    consume ((
        (p/i/n).exists() or _raise(FileNotFoundError(p/i/n))
        for i in input_dirs for n in image_names
    ))
    xy_name_pairs = [(p/i/n, p/output_dir/n) for i in input_dirs for n in image_names]
    n_images = len(xy_name_pairs)
    description = '{p}: output/target=\'{o}\' inputs={i}'.format(p=basepath,i=list(input_dirs),o=output_dir)

    def _gen():
        for fx, fy in xy_name_pairs:
            x, y = imread(str(fx)), imread(str(fy))
            # x,y = x[:,256:-256,256:-256],y[:,256:-256,256:-256] #tmp
            x.shape == y.shape or _raise(ValueError())
            yield x, y

    return InputData(_gen, n_images, description)



## Patch filter

def patch_filter_max(predicate=(lambda image,filtered: filtered > 0.4*np.percentile(image,99.9)), filter_size=None):
    from scipy.ndimage.filters import maximum_filter
    def _filter(image, patch_size, dtype=np.float32):
        if dtype is not None:
            image = image.astype(dtype)
        return predicate(image, maximum_filter(image, (patch_size if filter_size is None else filter_size), mode='constant'))
    return _filter


## Sample patches

def sample_patches_from_multiple_stacks(datas, patch_size, n_samples, patch_filter=None, verbose=False):
    """
    sample matching patches of size patch_size from all arrays in data

    patch_filter is applied to datas[0]

    signature: patch_filter(data, patch_size)
    """

    len(patch_size)==datas[0].ndim or _raise(ValueError())

    if not all(( a.shape == datas[0].shape for a in datas )):
        raise ValueError("all input shapes must be the same: %s" % (" / ".join(str(a.shape) for a in datas)))

    if not all(( 0 < s <= d for s,d in zip(patch_size,datas[0].shape) )):
        raise ValueError("patch_size %s negative or larger than data shape %s along some dimensions" % (str(patch_size), str(datas[0].shape)))

    if patch_filter is None:
        filter_mask = np.ones(datas[0].shape,dtype=np.bool)
    else:
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

    # res = [np.stack([data[r[0] - patch_size[0] // 2:r[0] + patch_size[0] - patch_size[0] // 2,
    #                  r[1] - patch_size[1] // 2:r[1] + patch_size[1] - patch_size[1] // 2,
    #                  r[2] - patch_size[2] // 2:r[2] + patch_size[2] - patch_size[2] // 2,
    #                  ] for r in zip(*rand_inds)]) for data in datas]

    # FIXME: Test this
    res = [np.stack([data[tuple(slice(_r-(_p//2),_r+_p-(_p//2)) for _r,_p in zip(r,patch_size))] for r in zip(*rand_inds)]) for data in datas]

    return res






## Crate training data

def create_patches (
        input_data,
        patch_size=(8,32,32),
        n_samples=2, # per image
        # transforms=[tf_flip(0)],
        transforms=None,
        patch_filter=patch_filter_max(),
        # patch_filter=None,
        percentiles=(1,99),
        # percentiles=lambda: (np.random.uniform(0,4), np.random.uniform(99.4,99.9)),
        percentile_axes=None,
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
    image_pairs, n_raw_images = input_data.generator(), input_data.size
    tf = Transform(*zip(*transforms)) # convert list of Transforms into Transform of lists
    image_pairs = compose(*tf.generator)(image_pairs) # combine all transformations with raw images as input
    n_transforms = np.prod(tf.size)
    n_images = n_raw_images * n_transforms
    n_patches = n_images * n_samples

    ## summary
    if verbose:
        print('='*66)
        print('%5d raw images x %4d transformations   = %5d images' % (n_raw_images,n_transforms,n_images))
        print('%5d images     x %4d patches per image = %5d patches in total' % (n_images,n_samples,n_patches))
        print('='*66)
        print('Input data:')
        print(input_data.description)
        print('='*66)
        print('Transformations:')
        for t in transforms:
            print('{t.size} x {t.name}'.format(t=t))
        print('='*66)
        print(flush=True)

    ## sample patches from each pair of transformed raw images
    X = np.empty((n_patches,)+tuple(patch_size),dtype=np.float32)
    Y = np.empty_like(X)

    for i,(x,y) in tqdm(enumerate(image_pairs),total=n_images):

        _Y,_X = sample_patches_from_multiple_stacks((y,x), patch_size, n_samples, patch_filter)
        s = slice(i*n_samples,(i+1)*n_samples)
        pmins, pmaxs = zip(*(norm_percentiles() for _ in range(n_samples)))
        X[s] = normalize_mi_ma(_X, np.percentile(x,pmins,axis=percentile_axes,keepdims=True),
                                   np.percentile(x,pmaxs,axis=percentile_axes,keepdims=True))
        Y[s] = normalize_mi_ma(_Y, np.min(y),
                                   np.percentile(y,pmaxs,axis=percentile_axes,keepdims=True))

        # del x,y
        # break

    if shuffle:
        shuffle_inplace(X,Y)

    return X,Y














from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

import numpy as np
from tifffile import imread
from collections import namedtuple

from tqdm import tqdm
from .utils import Path, normalize_mi_ma, _raise, consume, compose, shuffle_inplace


## Transforms (to be added later)

class Transform(namedtuple('Transform',('name','generator','size'))):
    """TODO"""

    def identity():
        """TODO"""
        def _gen(inputs):
            for d in inputs:
                yield d
        return Transform('Identity', _gen, 1)

    # def flip(axis):
    #     """TODO"""
    #     def _gen(inputs):
    #         for x,y,m_in in inputs:
    #             axis < x.ndim or _raise(ValueError())
    #             yield x, y, m_in
    #             yield np.flip(x,axis), np.flip(y,axis), None if m_in is None else np.flip(m_in,axis)
    #     return Transform('Flip (axis=%d)'%axis, _gen, 2)



## Input data

class InputData(namedtuple('InputData',('generator','size','description'))):
    """TODO"""

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
        `InputData` object, whose `generator` is used to yield all matching TIFF pairs.
        **Important**: the generator will return a tuple `(x,y,mask)`, where `x` is from
        `input_dirs` and `y` is the corresponding image from the `output_dir`; `mask` is
        set to `None`.

    Examples
    --------
    >>> !tree data
    data
    ├── GT
    │   ├── imageA.tif
    │   └── imageB.tif
    ├── input1
    │   ├── imageA.tif
    │   └── imageB.tif
    └── input2
        ├── imageA.tif
        └── imageB.tif

    >>> input_data = get_tiff_pairs_from_folders(basepath='data', input_dirs=['input1','input2'], output_dir='GT')
    >>> n_images = input_data.size
    >>> for input_x, output_y, mask in input_data.generator:
    >>>     ...

    Raises
    ------
    FileNotFoundError
        If an image found in `output_dir` does not exist in all `input_dirs`.
    ValueError
        If corresponding images do not have the same size (raised by returned InputData.generator).
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
            yield x, y, None

    return InputData(_gen, n_images, description)



## Patch filter

def no_background_patches(threshold=0.4, percentile=99.9):
    """Returns a patch filter to be used for :obj:`create_patches` to determine for each image pair which patches
    are eligible for sampling. The purpose is to only sample patches from "interesting" regions of the raw image that
    actually contain some non-background signal. To that end, a maximum filter is applied to the output/target image
    to find the largest values in a region.

    Parameters
    ----------
    threshold : float, optional
        Scalar threshold between 0 and 1 that will be multiplied with the (outlier-robust)
        maximum of the image (see `percentile` below) to denote a lower bound.
        Only patches with a maximum value above this lower bound are eligible to be sampled.
    percentile : float, optional
        Percentile value to denote the (outlier-robust) maximum of an image, i.e. should be close 100.

    Returns
    -------
    function
        Function that takes an image pair (y,x) and the patch size as arguments and
        returns a binary mask of the same size as the image (to denote the locations
        eligible for sampling for :obj:`create_patches`). At least one pixel of the
        binary mask must be True, otherwise there are no patches to sample.
    """

    (np.isscalar(percentile) and 0 <= percentile <= 100) or _raise(ValueError())
    (np.isscalar(threshold)  and 0 <= threshold  <= 1)   or _raise(ValueError())

    from scipy.ndimage.filters import maximum_filter
    def _filter(datas, patch_size, dtype=np.float32):
        image = datas[0]
        if dtype is not None:
            image = image.astype(dtype)
        filtered = maximum_filter(image, patch_size, mode='constant')
        return filtered > threshold * np.percentile(image,percentile)
    return _filter



## Sample patches

def sample_patches_from_multiple_stacks(datas, patch_size, n_samples, datas_mask=None, patch_filter=None, verbose=False):
    """ TODO: sample matching patches of size `patch_size` from all arrays in `datas` """

    len(patch_size)==datas[0].ndim or _raise(ValueError())

    if not all(( a.shape == datas[0].shape for a in datas )):
        raise ValueError("all input shapes must be the same: %s" % (" / ".join(str(a.shape) for a in datas)))

    if not all(( 0 < s <= d for s,d in zip(patch_size,datas[0].shape) )):
        raise ValueError("patch_size %s negative or larger than data shape %s along some dimensions" % (str(patch_size), str(datas[0].shape)))

    if patch_filter is None:
        patch_mask = np.ones(datas[0].shape,dtype=np.bool)
    else:
        patch_mask = patch_filter(datas, patch_size)

    if datas_mask is not None:
        # FIXME: Test this
        import warnings
        warnings.warn('Using pixel masks for raw/transformed images not tested.')
        datas_mask.shape == datas[0].shape or _raise(ValueError())
        datas_mask.dtype == np.bool or _raise(ValueError())
        from scipy.ndimage.filters import minimum_filter
        patch_mask &= minimum_filter(datas_mask, patch_size, mode='constant', cval=False)

    # get the valid indices

    border_slices = tuple([slice(s // 2, d - s + s // 2 + 1) for s, d in zip(patch_size, datas[0].shape)])
    valid_inds = np.where(patch_mask[border_slices])

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

def _valid_low_high_percentiles(ps):
    return isinstance(ps,(list,tuple,np.ndarray)) and len(ps)==2 and all(map(np.isscalar,ps)) and (0<=ps[0]<ps[1]<=100)

def sample_percentiles(pmin=(1,4), pmax=(99.4,99.9)):
    """ TODO """
    _valid_low_high_percentiles(pmin) or _raise(ValueError(pmin))
    _valid_low_high_percentiles(pmax) or _raise(ValueError(pmax))
    pmin[1] < pmax[0] or _raise(ValueError())
    return lambda: (np.random.uniform(*pmin), np.random.uniform(*pmax))

def create_patches (
        input_data,
        patch_size,
        n_samples,
        transforms = None,
        patch_filter = no_background_patches(),
        # percentiles = (1,99),
        percentiles = sample_percentiles(),
        percentile_axes = None,
        shuffle = True,
        verbose = True,
    ):
    """Create normalized training data to be used for neural network training.

    Parameters
    ----------
    input_data : :obj:`InputData`
        Object that yields matching pairs of raw images.
    patch_size : tuple of int
        Shape of the patches to be extraced from raw images.
        Must be compatible with the number of dimensions (2D/3D) and the shape of the raw images.
    n_samples : int
        Number of patches to be sampled/extracted from each raw image pair (after transformations, see below).
    transforms : iterable of :obj:`Transform`, optional
        List of `Transform` objects that apply additional transformations to the raw images.
        This can be used to augment the set of raw images (e.g., by including rotations).
    patch_filter : function, optional
        Function to determine for each image pair which patches are eligible to be extracted.
        See :obj:`no_background_patches`.
    percentiles : tuple of two ints, or function that returns tuple of two ints
        A tuple (`pmin`, `pmax`) or a function that returns such a tuple, where the extracted patches
        are (affinely) normalized in such that a value of 0 (1) corresponds to the `pmin`-th (`pmax`-th) percentile
        of the raw image. We recommend to sample from a range of percentile values (see :obj:`sample_percentiles`).
    percentile_axes : tuple of ints or None, optional
        List of axis over which the percentiles are computed.
        Only necessary for multi-channel images, where each channel may be normalized independently.
    shuffle : bool, optional
        Randomly shuffle all extracted patches.
    verbose : bool, optional
        Display overview of images, transforms, etc.

    Returns
    -------
    `tuple` of :obj:`np.ndarray`
        Returns a pair (`X`, `Y`) of arrays with the normalized extracted patches from all (transformed) raw images.
        `X` is the array of patches extracted from input images with `Y` being the array of corresponding output patches.

    Examples
    --------
    >>> input_data = get_tiff_pairs_from_folders(basepath='data', input_dirs=['input1','input2'], output_dir='GT')
    >>> X, Y = create_patches(input_data, patch_size=(32,128,128), n_samples=16,
                              percentiles = lambda: (np.random.uniform(1,4), np.random.uniform(99.4,99.9)))

    Raises
    ------
    ValueError
        Various reasons.
    """

    ## percentiles
    if callable(percentiles):
        _p_example = percentiles()
        _valid_low_high_percentiles(_p_example) or _raise(ValueError(_p_example))
        norm_percentiles = percentiles
    else:
        _valid_low_high_percentiles(percentiles) or _raise(ValueError(percentiles))
        norm_percentiles = lambda: percentiles

    ## images and transforms
    if transforms is None or len(transforms)==0:
        transforms = (Transform.identity(),)
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

    for i, (x,y,mask) in tqdm(enumerate(image_pairs),total=n_images):

        _Y,_X = sample_patches_from_multiple_stacks((y,x), patch_size, n_samples, mask, patch_filter)
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

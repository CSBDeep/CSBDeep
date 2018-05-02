from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

import numpy as np
from tifffile import imread
from collections import namedtuple
import sys, os, warnings

from tqdm import tqdm
from .utils import Path, normalize_mi_ma, _raise, consume, compose, shuffle_inplace, axes_dict, move_image_axes


## Transforms (to be added later)

class Transform(namedtuple('Transform',('name','generator','size'))):
    """Extension of :func:`collections.namedtuple` with three fields: `name`, `generator`, and `size`.

    Parameters
    ----------
    name : str
        Name of the applied transformation.
    generator : function
        Function that takes a generator as input and itself returns a generator; input and returned
        generator have the same structure as that of :class:`RawData`.
        The purpose of the returned generator is to augment the images provided by the input generator
        through additional transformations.
        It is important that the returned generator also includes every input tuple unchanged.
    size : int
        Number of transformations applied to every image (obtained from the input generator).
    """

    def identity():
        """
        Returns
        -------
        Transform
            Identity transformation that passes every input through unchanged.
        """
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



## Raw data

class RawData(namedtuple('RawData',('generator','size','description'))):
    """:func:`collections.namedtuple` with three fields: `generator`, `size`, and `description`.

    Parameters
    ----------
    generator : function
        Function without arguments that returns a generator that yields tuples `(x,y,mask)`,
        where `x` is a source image (e.g., with low SNR) with `y` being the corresponding target image
        (e.g., with high SNR); `mask` can either be `None` or a boolean array that denotes which
        pixels are eligible to extracted in :func:`create_patches`. Note that `x`, `y`, and `mask`
        must all be of type :class:`numpy.ndarray` with the same shape.
    size : int
        Number of tuples that the `generator` will yield.
    description : str
        Textual description of the raw data.
    """

def get_tiff_pairs_from_folders(basepath,source_dirs,target_dir,axes='CZYX',pattern='*.tif*'):
    """Get pairs of corresponding TIFF images read from folders.

    Two images correspond to each other if they have the same file name, but are located in different folders.

    Parameters
    ----------
    basepath : str
        Base folder that contains sub-folders with images.
    source_dirs : list or tuple
        List of folder names relative to `basepath` that contain the source images (e.g., with low SNR).
    target_dir : str
        Folder name relative to `basepath` that contains the target images (e.g., with high SNR).
    axes : str
        Semantics of axes of loaded images (must be same for all images).
    pattern : str
        Glob-style pattern to match the desired TIFF images.

    Returns
    -------
    RawData
        :obj:`RawData` object, whose `generator` is used to yield all matching TIFF pairs.
        The generator will return a tuple `(x,y,axes,mask)`, where `x` is from
        `source_dirs` and `y` is the corresponding image from the `target_dir`;
        `mask` is set to `None`.

    Raises
    ------
    FileNotFoundError
        If an image found in `target_dir` does not exist in all `source_dirs`.
    ValueError
        If corresponding images do not have the same size (raised by returned :func:`RawData.generator`).

    Example
    --------
    >>> !tree data
    data
    ├── GT
    │   ├── imageA.tif
    │   └── imageB.tif
    ├── source1
    │   ├── imageA.tif
    │   └── imageB.tif
    └── source2
        ├── imageA.tif
        └── imageB.tif

    >>> data = get_tiff_pairs_from_folders(basepath='data', source_dirs=['source1','source2'], target_dir='GT', axes='YX')
    >>> n_images = data.size
    >>> for source_x, target_y, axes, mask in data.generator():
    ...     pass
    """

    p = Path(basepath)
    image_names = [f.name for f in (p/target_dir).glob(pattern)]
    len(image_names) > 0 or _raise(FileNotFoundError("'target_dir' doesn't exist or didn't find any images in it."))
    consume ((
        (p/s/n).exists() or _raise(FileNotFoundError(p/s/n))
        for s in source_dirs for n in image_names
    ))
    isinstance(axes,str) or _raise(ValueError())
    axes = axes.upper()
    xy_name_pairs = [(p/source_dir/n, p/target_dir/n) for source_dir in source_dirs for n in image_names]
    n_images = len(xy_name_pairs)
    description = '{p}: target=\'{o}\', sources={s}, axes={a}, pattern={pt}'.format(p=basepath,s=list(source_dirs),o=target_dir,a=axes,pt=pattern)

    def _gen():
        for fx, fy in xy_name_pairs:
            x, y = imread(str(fx)), imread(str(fy))
            # x,y = x[:,256:-256,256:-256],y[:,256:-256,256:-256] #tmp
            x.shape == y.shape or _raise(ValueError())
            len(axes) >= x.ndim or _raise(ValueError())
            yield x, y, axes[-x.ndim:], None

    return RawData(_gen, n_images, description)



## Patch filter

def no_background_patches(threshold=0.4, percentile=99.9):
    """Returns a patch filter to be used by :func:`create_patches` to determine for each image pair which patches
    are eligible for sampling. The purpose is to only sample patches from "interesting" regions of the raw image that
    actually contain some non-background signal. To that end, a maximum filter is applied to the target image
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
        Function that takes an image pair `(y,x)` and the patch size as arguments and
        returns a binary mask of the same size as the image (to denote the locations
        eligible for sampling for :func:`create_patches`). At least one pixel of the
        binary mask must be ``True``, otherwise there are no patches to sample.

    Raises
    ------
    ValueError
        Illegal arguments.
    """

    (np.isscalar(percentile) and 0 <= percentile <= 100) or _raise(ValueError())
    (np.isscalar(threshold)  and 0 <= threshold  <=   1) or _raise(ValueError())

    from scipy.ndimage.filters import maximum_filter
    def _filter(datas, patch_size, dtype=np.float32):
        image = datas[0]
        if dtype is not None:
            image = image.astype(dtype)
        # make max filter patch_size smaller to avoid only few non-bg pixel close to image border
        patch_size = [(p//2 if p>1 else p) for p in patch_size]
        filtered = maximum_filter(image, patch_size, mode='constant')
        return filtered > threshold * np.percentile(image,percentile)
    return _filter



## Sample patches

def sample_patches_from_multiple_stacks(datas, patch_size, n_samples, datas_mask=None, patch_filter=None, verbose=False):
    """ sample matching patches of size `patch_size` from all arrays in `datas` """

    # TODO: some of these checks are already required in 'create_patches'
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

def _memory_check(n_required_memory_bytes, thresh_free_frac=0.5, thresh_abs_bytes=1024*1024**2):
    try:
        # raise ImportError
        import psutil
        mem = psutil.virtual_memory()
        mem_frac = n_required_memory_bytes / mem.available
        if mem_frac > 1:
            raise MemoryError('Not enough available memory.')
        elif mem_frac > thresh_free_frac:
            print('Warning: will use at least %.0f MB (%.1f%%) of available memory.\n' % (n_required_memory_bytes/1024**2,100*mem_frac), file=sys.stderr, flush=True)
    except ImportError:
        if n_required_memory_bytes > thresh_abs_bytes:
            print('Warning: will use at least %.0f MB of memory.\n' % (n_required_memory_bytes/1024**2), file=sys.stderr, flush=True)

def sample_percentiles(pmin=(1,3), pmax=(99.5,99.9)):
    """Sample percentile values from a uniform distribution.

    Parameters
    ----------
    pmin : tuple
        Tuple of two values that denotes the interval for sampling low percentiles.
    pmax : tuple
        Tuple of two values that denotes the interval for sampling high percentiles.

    Returns
    -------
    function
        Function without arguments that returns `(pl,ph)`, where `pl` (`ph`) is a sampled low (high) percentile.

    Raises
    ------
    ValueError
        Illegal arguments.
    """
    _valid_low_high_percentiles(pmin) or _raise(ValueError(pmin))
    _valid_low_high_percentiles(pmax) or _raise(ValueError(pmax))
    pmin[1] < pmax[0] or _raise(ValueError())
    return lambda: (np.random.uniform(*pmin), np.random.uniform(*pmax))


def norm_percentiles(percentiles=sample_percentiles(), relu_last=False):
    """Normalize extracted patches based on percentiles from corresponding raw image.

    Parameters
    ----------
    percentiles : tuple, optional
        A tuple (`pmin`, `pmax`) or a function that returns such a tuple, where the extracted patches
        are (affinely) normalized in such that a value of 0 (1) corresponds to the `pmin`-th (`pmax`-th) percentile
        of the raw image (default: :func:`sample_percentiles`).
    relu_last : bool, optional
        Flag to indicate whether the last activation of the CARE network is/will be using
        a ReLU activation function (default: ``False``)

    Return
    ------
    function
        Function that does percentile-based normalization to be used in :func:`create_patches`.

    Raises
    ------
    ValueError
        Illegal arguments.

    Todo
    ----
    ``relu_last`` flag problematic/inelegant.

    """
    if callable(percentiles):
        _tmp = percentiles()
        _valid_low_high_percentiles(_tmp) or _raise(ValueError(_tmp))
        get_percentiles = percentiles
    else:
        _valid_low_high_percentiles(percentiles) or _raise(ValueError(percentiles))
        get_percentiles = lambda: percentiles

    def _normalize(patches_x,patches_y, x,y,mask,channel):
        pmins, pmaxs = zip(*(get_percentiles() for _ in patches_x))
        percentile_axes = None if channel is None else tuple((d for d in range(x.ndim) if d != channel))
        _perc = lambda a,p: np.percentile(a,p,axis=percentile_axes,keepdims=True)
        patches_x_norm = normalize_mi_ma(patches_x, _perc(x,pmins), _perc(x,pmaxs))
        if relu_last:
            pmins = np.zeros_like(pmins)
        patches_y_norm = normalize_mi_ma(patches_y, _perc(y,pmins), _perc(y,pmaxs))
        return patches_x_norm, patches_y_norm

    return _normalize


def create_patches(
        raw_data,
        patch_size,
        n_patches_per_image,
        patch_axes    = None,
        save_file     = None,
        transforms    = None,
        patch_filter  = no_background_patches(),
        normalization = norm_percentiles(),
        shuffle       = True,
        verbose       = True,
    ):
    """Create normalized training data to be used for neural network training.

    Parameters
    ----------
    raw_data : :class:`RawData`
        Object that yields matching pairs of raw images.
    patch_size : tuple
        Shape of the patches to be extraced from raw images.
        Must be compatible with the number of dimensions and axes of the raw images.
    n_patches_per_image : int
        Number of patches to be sampled/extracted from each raw image pair (after transformations, see below).
    patch_axes : str or None
        Axes of the extracted patches. If ``None``, will assume to be equal to that of transformed raw data.
    save_file : str or None
        File name to save training data to disk in ``.npz`` format (see :func:`numpy.savez`).
        If ``None``, data will not be saved.
    transforms : list or tuple, optional
        List of :class:`Transform` objects that apply additional transformations to the raw images.
        This can be used to augment the set of raw images (e.g., by including rotations).
        Set to ``None`` to disable. Default: ``None``.
    patch_filter : function, optional
        Function to determine for each image pair which patches are eligible to be extracted
        (default: :func:`no_background_patches`). Set to ``None`` to disable.
    normalization : function, optional
        Function that takes arguments `(patches_x, patches_y, x, y, mask, channel)`, whose purpose is to
        normalize the patches (`patches_x`, `patches_y`) extracted from the associated raw images
        (`x`, `y`, with `mask`; see :class:`RawData`). Default: :func:`norm_percentiles`.
    shuffle : bool, optional
        Randomly shuffle all extracted patches.
    verbose : bool, optional
        Display overview of images, transforms, etc.

    Returns
    -------
    tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`, str)
        Returns a tuple (`X`, `Y`, `axes`) with the normalized extracted patches from all (transformed) raw images
        and their axes.
        `X` is the array of patches extracted from source images with `Y` being the array of corresponding target patches.
        The shape of `X` and `Y` is as follows: `(n_total_patches, n_channels, ...)`.
        For single-channel images (`channel` = ``None``), `n_channels` = 1.

    Raises
    ------
    ValueError
        Various reasons.

    Example
    -------
    >>> raw_data = get_tiff_pairs_from_folders(basepath='data', source_dirs=['source1','source2'], target_dir='GT', axes='ZYX')
    >>> X, Y, XY_axes = create_patches(raw_data, patch_size=(32,128,128), n_patches_per_image=16)

    Todo
    ----
    - Is :func:`create_patches` a good name?
    - Save created patches directly to disk using :class:`numpy.memmap` or similar?
      Would allow to work with large data that doesn't fit in memory.

    """
    ## images and transforms
    if transforms is None:
        transforms = []
    transforms = list(transforms)
    if patch_axes is not None:
        transforms.append(permute_axes(patch_axes))
    if len(transforms) == 0:
        transforms.append(Transform.identity())


    image_pairs, n_raw_images = raw_data.generator(), raw_data.size
    tf = Transform(*zip(*transforms)) # convert list of Transforms into Transform of lists
    image_pairs = compose(*tf.generator)(image_pairs) # combine all transformations with raw images as input
    n_transforms = np.prod(tf.size)
    n_images = n_raw_images * n_transforms
    n_patches = n_images * n_patches_per_image
    n_required_memory_bytes = 2 * n_patches*np.prod(patch_size) * 4

    save_file is None or isinstance(save_file,str) or _raise(ValueError())
    if save_file is not None:
        # append .npz suffix
        if os.path.splitext(save_file)[1] != '.npz':
            save_file += '.npz'

    ## memory check
    _memory_check(n_required_memory_bytes)

    ## summary
    if verbose:
        print('='*66)
        print('%5d raw images x %4d transformations   = %5d images' % (n_raw_images,n_transforms,n_images))
        print('%5d images     x %4d patches per image = %5d patches in total' % (n_images,n_patches_per_image,n_patches))
        print('='*66)
        print('Input data:')
        print(raw_data.description)
        print('='*66)
        print('Transformations:')
        for t in transforms:
            print('{t.size} x {t.name}'.format(t=t))
        print('='*66)
        print(flush=True)

    ## sample patches from each pair of transformed raw images
    X = np.empty((n_patches,)+tuple(patch_size),dtype=np.float32)
    Y = np.empty_like(X)

    for i, (x,y,_axes,mask) in tqdm(enumerate(image_pairs),total=n_images):
        if i==0:
            axes = _axes.upper()
            channel = axes_dict(axes)['C']
        # checks
        (isinstance(axes,str) and len(axes) >= x.ndim) or _raise(ValueError())
        axes == _axes.upper() or _raise(ValueError('not all images have the same axes.'))
        x.shape == y.shape or _raise(ValueError())
        mask is None or mask.shape == x.shape or _raise(ValueError())
        (channel is None or (isinstance(channel,int) and 0<=channel<x.ndim)) or _raise(ValueError())
        channel is None or patch_size[channel]==x.shape[channel] or _raise(ValueError('extracted patches must contain all channels.'))

        _Y,_X = sample_patches_from_multiple_stacks((y,x), patch_size, n_patches_per_image, mask, patch_filter)

        s = slice(i*n_patches_per_image,(i+1)*n_patches_per_image)
        X[s], Y[s] = normalization(_X,_Y, x,y,mask,channel)

    if shuffle:
        shuffle_inplace(X,Y)

    axes = 'SC'+axes.replace('C','')
    if channel is None:
        X = np.expand_dims(X,1)
        Y = np.expand_dims(Y,1)
    else:
        X = np.moveaxis(X, 1+channel, 1)
        Y = np.moveaxis(Y, 1+channel, 1)

    if save_file is not None:
        print('Saving data to %s.' % save_file)
        np.savez(save_file, X=X, Y=Y, axes=axes)

    return X,Y,axes



def anisotropic_distortions(
        subsample,
        psf,
        psf_axes       = None,
        poisson_noise  = False,
        gauss_sigma    = 0,
        crop_threshold = 0.2,
    ):
    """Simulate anisotropic distortions.

    Modify image along X axis to mimic the distortions that occur due to
    low resolution along Z axis. Note that the modified image is finally upscaled
    to obtain the same resolution as the unmodified input image.

    The following operations are applied to the image (in order):

    1. Convolution with PSF
    2. Poisson noise
    3. Gaussian noise
    4. Subsampling along X axis
    5. Upsampling along X axis (to former size).


    Parameters
    ----------
    subsample : float
        Subsampling factor to apply to X axis to mimic distortions along Z.
    psf : :class:`numpy.ndarray` or None
        Point spread function (PSF) that is supposed to mimic blurring
        of the microscope due to reduced axial resolution.
        Must be compatible with the number of dimensions (2D/3D) and the shape of the raw images.
        Set to ``None`` to disable.
    psf_axes : str or None
        Axes of the psf. If ``None``, psf axes are assumed to be the same as of the image
        that it is applied to.
    poisson_noise : bool
        Flag to indicate whether Poisson noise should be added to the image.
    gauss_sigma : float
        Standard deviation of white Gaussian noise to be added to the image.
    crop_threshold : float
        The subsample factor must evenly divide the image size along the X axis to prevent
        potential image misalignment. If this is not the case the subsample factors are
        modified and the raw image will be cropped along X up to a fraction indicated by `crop_threshold`.

    Returns
    -------
    Transform
        Returns a :class:`Transform` object to be used with :func:`create_patches` to
        create training data for an isotropic reconstruction CARE network.

    Raises
    ------
    ValueError
        Various reasons.

    """
    zoom_order = 1

    (np.isscalar(subsample) and subsample >= 1) or _raise(ValueError('subsample must be >= 1'))
    _subsample = subsample


    psf is None or isinstance(psf,np.ndarray) or _raise(ValueError())
    psf_axes is None or isinstance(psf_axes,str) or _raise(ValueError())

    0 < crop_threshold < 1 or _raise(ValueError())


    def _make_normalize_data(axes_in,axes_out='XY'):
        """Move X to front of image."""
        ax = axes_dict(axes_in)
        all((ax[a] is not None) for a in axes_out) or _raise(ValueError('X and/or Y axes missing.'))
        # add axis in axes_in to axes_out (if it doesn't exist there)
        for a in ax:
            if (ax[a] is not None) and (a not in axes_out):
                axes_out += a

        def _normalize_data(data,undo=False):
            if undo:
                return move_image_axes(data, axes_out, axes_in)
            else:
                return move_image_axes(data, axes_in, axes_out)
        return _normalize_data


    def _scale_down_up(data,subsample):
        from scipy.ndimage.interpolation import zoom
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            factor = np.ones(data.ndim)
            factor[0] = subsample
            return zoom(zoom(data, 1/factor, order=0),
                                     factor, order=zoom_order)


    def _adjust_subsample(d,s,c):
        """length d, subsample s, tolerated crop loss fraction c"""
        from fractions import Fraction

        def crop_size(n_digits,frac):
            _s = round(s,n_digits)
            _div = frac.denominator
            s_multiple_max = np.floor(d/_s)
            s_multiple = (s_multiple_max//_div)*_div
            # print(n_digits, _s,_div,s_multiple)
            size = s_multiple * _s
            assert np.allclose(size,round(size))
            return size

        def decimals(v,n_digits=None):
            if n_digits is not None:
                v = round(v,n_digits)
            s = str(v)
            assert '.' in s
            decimals = s[1+s.find('.'):]
            return int(decimals), len(decimals)

        s = float(s)
        dec, n_digits = decimals(s)
        frac = Fraction(dec,10**n_digits)
        # a multiple of s that is also an integer number must be
        # divisible by the denominator of the fraction that represents the decimal points

        # round off decimals points if needed
        while n_digits > 0 and (d-crop_size(n_digits,frac))/d > c:
            n_digits -= 1
            frac = Fraction(decimals(s,n_digits)[0], 10**n_digits)

        size = crop_size(n_digits,frac)
        if size == 0 or (d-size)/d > c:
            raise ValueError("subsample factor %g too large (crop_threshold=%g)" % (s,c))

        return round(s,n_digits), int(round(crop_size(n_digits,frac)))


    def _make_divisible_by_subsample(x,size):
        def _split_slice(v):
            return slice(None) if v==0 else slice(v//2,-(v-v//2))
        slices = [slice(None) for _ in x.shape]
        slices[0] = _split_slice(x.shape[0]-size)
        return x[slices]


    def _generator(inputs):
        for img,y,axes,mask in inputs:

            if not (y is None or np.all(img==y)):
                warnings.warn('ignoring y.')
            if mask is not None:
                warnings.warn('ignoring mask.')
            del y, mask


            # tmp
            # img = img[...,:256,:256]


            _normalize_data = _make_normalize_data(axes)
            # print(axes, img.shape)

            x = img.astype(np.float32, copy=False)

            if psf is not None:
                np.min(psf) >= 0 or _raise(ValueError('psf has negative values.'))
                _psf = psf / np.sum(psf)
                x.ndim == _psf.ndim or _raise(ValueError('image and psf must have the same number of dimensions.'))
                if psf_axes is not None:
                    sorted(axes) == sorted(psf_axes) or _raise(ValueError('psf_axes (%s) not compatible with that of the image (%s)' % (psf_axes,axes)))
                    _psf = move_image_axes(_psf, psf_axes, axes)
                # print("blurring with psf")
                from scipy.signal import fftconvolve
                x = fftconvolve(x, _psf.astype(np.float32,copy=False), mode='same')

            if bool(poisson_noise):
                # print("apply poisson noise")
                x = np.random.poisson(np.maximum(0,x).astype(np.int)).astype(np.float32)

            if gauss_sigma > 0:
                # print("adding gaussian noise with sigma = ", gauss_sigma)
                noise = np.random.normal(0,gauss_sigma,size=x.shape).astype(np.float32)
                x = np.maximum(0,x+noise)

            if _subsample != 1:
                # print("down and upsampling X by factor %s" % str(_subsample))
                img = _normalize_data(img)
                x   = _normalize_data(x)

                subsample, subsample_size = _adjust_subsample(x.shape[0],_subsample,crop_threshold)
                # print(subsample, subsample_size)
                if _subsample != subsample:
                    warnings.warn('changing subsample from %s to %s' % (str(_subsample),str(subsample)))

                img = _make_divisible_by_subsample(img,subsample_size)
                x   = _make_divisible_by_subsample(x,  subsample_size)
                x   = _scale_down_up(x,subsample)

                assert x.shape == img.shape, (x.shape, img.shape)

                img = _normalize_data(img,undo=True)
                x   = _normalize_data(x,  undo=True)

            yield x, img, axes, None


    return Transform('Anisotropic distortion (along X axis)', _generator, 1)






def permute_axes(axes):
    """Transformation to permute images axes.

    Note that input images must have compatible axes, i.e.
    they must be a permutation of the target axis.

    Parameters
    ----------
    axes : str
        Target axis, to which the input images will be permuted.

    Returns
    -------
    Transform
        Returns a :class:`Transform` object whose `generator` will
        perform the axes permutation of `x`, `y`, and `mask`.

    """
    axes = str(axes).upper()
    axes_dict(axes) # does some error checking

    def _generator(inputs):
        for x, y, axes_in, mask in inputs:
            axes_in = str(axes_in).upper()
            if axes_in != axes:
                # print('permuting axes from %s to %s' % (axes_in,axes))
                x = move_image_axes(x, axes_in, axes)
                y = move_image_axes(y, axes_in, axes)
                if mask is not None:
                    mask = move_image_axes(mask, axes_in, axes)
            yield x, y, axes, mask

    return Transform('Permute axes to %s' % axes, _generator, 1)
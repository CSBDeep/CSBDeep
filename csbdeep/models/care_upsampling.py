from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
from scipy.ndimage.interpolation import zoom

from csbdeep.internals.probability import ProbabilisticPrediction
from .care_standard import CARE
from ..internals.predict import PercentileNormalizer, PadAndCropResizer
from ..utils import _raise, axes_dict


class UpsamplingCARE(CARE):
    """CARE network for image reconstruction with undersampled Z dimension.
    TODO: this is different from IsotropicCARE... needs explanation!

    Extends :class:`csbdeep.models.CARE` by replacing prediction
    (:func:`predict`, :func:`predict_probabilistic`) to first upscale Z before image restoration.
    """

    def predict(self, img, axes, factor, normalizer=PercentileNormalizer(), resizer=PadAndCropResizer(), n_tiles=1):
        """Apply neural network to raw image with undersampled Z resolution.

        Parameters
        ----------
        img : :class:`numpy.ndarray`
            Raw input image, with image dimensions expected in the same order as in data for training.
            If applicable, only the z and channel dimensions can be anywhere.
            TODO: docstrings need update now that "axes" is used.
        axes : str
            Axes of ``img``.
        factor : float
            Upsampling factor for z dimension. It is important that this is chosen in correspondence
            to the subsampling factor used during training data generation.
            TODO: 'factor' should be called 'subsample' to be consistent with training data generation?
        normalizer : :class:`csbdeep.predict.Normalizer`
            Normalization of input image before prediction and (potentially) transformation back after prediction.
        resizer : :class:`csbdeep.predict.Resizer`
            If necessary, input image is resized to enable neural network prediction and result is (possibly)
            resized to yield original image size.
        n_tiles : int
            Out of memory (OOM) errors can occur if the input image is too large.
            To avoid this problem, the input image is broken up into (overlapping) tiles
            that can then be processed independently and re-assembled to yield the restored image.
            This parameter denotes the number of tiles. Note that if the number of tiles is too low,
            it is adaptively increased until OOM errors are avoided, albeit at the expense of runtime.

        Returns
        -------
        :class:`numpy.ndarray`
            Returns the restored image. If the model is probabilistic, this denotes the `mean` parameter of
            the predicted per-pixel Laplace distributions (i.e., the expected restored image).
            Axes ordering is unchanged wrt input image. Only if there the output is multi-channel and
            the input image didn't have a channel axis, then channels are appended at the end.

        """
        img = self._upsample(img, axes, factor)
        return self._predict_mean_and_scale(img, axes, normalizer, resizer, n_tiles)[0]


    def predict_probabilistic(self, img, axes, factor, normalizer=PercentileNormalizer(), resizer=PadAndCropResizer(), n_tiles=1):
        """Apply neural network to raw image to predict probability distribution for image with undersampled Z resolution.

        See :func:`predict` for parameter explanations.

        Returns
        -------
        :class:`csbdeep.probability.ProbabilisticPrediction`
            Returns the probability distribution of the restored image.

        Raises
        ------
        ValueError
            If this is not a probabilistic model.

        """
        self.config.probabilistic or _raise(ValueError('This is not a probabilistic model.'))
        img = self._upsample(img, axes, factor)
        mean, scale = self._predict_mean_and_scale(img, axes, factor, normalizer, resizer, n_tiles)
        return ProbabilisticPrediction(mean, scale)


    @staticmethod
    def _upsample(img, axes, factor, axis='Z'):
        factors = np.ones(img.ndim)
        factors[axes_dict(axes)[axis]] = factor
        return zoom(img,factors,order=1)

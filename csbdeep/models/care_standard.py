# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, absolute_import, division

import warnings
import numpy as np
from six import string_types

from csbdeep.internals.probability import ProbabilisticPrediction
from .config import Config
from .base_model import BaseModel, suppress_without_basedir

from ..utils import _raise, axes_check_and_normalize, axes_dict, move_image_axes
from ..utils.six import Path
from ..utils.tf import export_SavedModel, IS_TF_1, keras_import, CARETensorBoardImage
from ..version import __version__ as package_version
from ..data import Normalizer, NoNormalizer, PercentileNormalizer
from ..data import Resizer, NoResizer, PadAndCropResizer
from ..internals.predict import predict_tiled, tile_overlap, Progress, total_n_tiles
from ..internals import nets, train

from packaging.version import Version
keras = keras_import()

import tensorflow as tf
# if IS_TF_1:
#     import tensorflow as tf
# else:
#     import tensorflow.compat.v1 as tf
#     # tf.disable_v2_behavior()


class CARE(BaseModel):
    """Standard CARE network for image restoration and enhancement.

    Uses a convolutional neural network created by :func:`csbdeep.internals.nets.common_unet`.
    Note that isotropic reconstruction and manifold extraction/projection are not supported here
    (see :class:`csbdeep.models.IsotropicCARE` ).

    Parameters
    ----------
    config : :class:`csbdeep.models.Config` or None
        Valid configuration of CARE network (see :func:`Config.is_valid`).
        Will be saved to disk as JSON (``config.json``).
        If set to ``None``, will be loaded from disk (must exist).
    name : str or None
        Model name. Uses a timestamp if set to ``None`` (default).
    basedir : str
        Directory that contains (or will contain) a folder with the given model name.
        Use ``None`` to disable saving (or loading) any data to (or from) disk (regardless of other parameters).

    Raises
    ------
    FileNotFoundError
        If ``config=None`` and config cannot be loaded from disk.
    ValueError
        Illegal arguments, including invalid configuration.

    Example
    -------
    >>> model = CARE(config, 'my_model')

    Attributes
    ----------
    config : :class:`csbdeep.models.Config`
        Configuration of CARE network, as provided during instantiation.
    keras_model : `Keras model <https://keras.io/getting-started/functional-api-guide/>`_
        Keras neural network model.
    name : str
        Model name.
    logdir : :class:`pathlib.Path`
        Path to model folder (which stores configuration, weights, etc.)
    """

    def __init__(self, config, name=None, basedir='.'):
        """See class docstring."""
        super(CARE, self).__init__(config=config, name=name, basedir=basedir)


    def _build(self):
        return nets.common_unet(
            n_dim           = self.config.n_dim,
            n_channel_out   = self.config.n_channel_out,
            prob_out        = self.config.probabilistic,
            residual        = self.config.unet_residual,
            n_depth         = self.config.unet_n_depth,
            kern_size       = self.config.unet_kern_size,
            n_first         = self.config.unet_n_first,
            last_activation = self.config.unet_last_activation,
        )(self.config.unet_input_shape)


    def prepare_for_training(self, optimizer=None, **kwargs):
        """Prepare for neural network training.

        Calls :func:`csbdeep.internals.train.prepare_model` and creates
        `Keras Callbacks <https://keras.io/callbacks/>`_ to be used for training.

        Note that this method will be implicitly called once by :func:`train`
        (with default arguments) if not done so explicitly beforehand.

        Parameters
        ----------
        optimizer : obj or None
            Instance of a `Keras Optimizer <https://keras.io/optimizers/>`_ to be used for training.
            If ``None`` (default), uses ``Adam`` with the learning rate specified in ``config``.
        kwargs : dict
            Additional arguments for :func:`csbdeep.internals.train.prepare_model`.

        """
        if optimizer is None:
            Adam = keras_import('optimizers', 'Adam')
            learning_rate = 'lr' if Version(keras.__version__) < Version('2.3.0') else 'learning_rate'
            optimizer = Adam(**{learning_rate: self.config.train_learning_rate})
        self.callbacks = train.prepare_model(self.keras_model, optimizer, self.config.train_loss, **kwargs)

        if self.basedir is not None:
            self.callbacks += self._checkpoint_callbacks()

            if self.config.train_tensorboard:
                if IS_TF_1:
                    from ..utils.tf import CARETensorBoard
                    self.callbacks.append(CARETensorBoard(log_dir=str(self.logdir), prefix_with_timestamp=False, n_images=3, write_images=True, prob_out=self.config.probabilistic))
                else:
                    from tensorflow.keras.callbacks import TensorBoard
                    self.callbacks.append(TensorBoard(log_dir=str(self.logdir/'logs'), write_graph=False, profile_batch=0))

        if self.config.train_reduce_lr is not None:
            ReduceLROnPlateau = keras_import('callbacks', 'ReduceLROnPlateau')
            rlrop_params = self.config.train_reduce_lr
            if 'verbose' not in rlrop_params:
                rlrop_params['verbose'] = True
            # TF2: add as first callback to put 'lr' in the logs for TensorBoard
            self.callbacks.insert(0,ReduceLROnPlateau(**rlrop_params))

        self._model_prepared = True


    def train(self, X,Y, validation_data, epochs=None, steps_per_epoch=None):
        """Train the neural network with the given data.

        Parameters
        ----------
        X : :class:`numpy.ndarray`
            Array of source images.
        Y : :class:`numpy.ndarray`
            Array of target images.
        validation_data : tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`)
            Tuple of arrays for source and target validation images.
        epochs : int
            Optional argument to use instead of the value from ``config``.
        steps_per_epoch : int
            Optional argument to use instead of the value from ``config``.

        Returns
        -------
        ``History`` object
            See `Keras training history <https://keras.io/models/model/#fit>`_.

        """
        ((isinstance(validation_data,(list,tuple)) and len(validation_data)==2)
            or _raise(ValueError('validation_data must be a pair of numpy arrays')))

        n_train, n_val = len(X), len(validation_data[0])
        frac_val = (1.0 * n_val) / (n_train + n_val)
        frac_warn = 0.05
        if frac_val < frac_warn:
            warnings.warn("small number of validation images (only %.1f%% of all images)" % (100*frac_val))
        axes = axes_check_and_normalize('S'+self.config.axes,X.ndim)
        ax = axes_dict(axes)

        for a,div_by in zip(axes,self._axes_div_by(axes)):
            n = X.shape[ax[a]]
            if n % div_by != 0:
                raise ValueError(
                    "training images must be evenly divisible by %d along axis %s"
                    " (which has incompatible size %d)" % (div_by,a,n)
                )

        if epochs is None:
            epochs = self.config.train_epochs
        if steps_per_epoch is None:
            steps_per_epoch = self.config.train_steps_per_epoch

        if not self._model_prepared:
            self.prepare_for_training()

        if (self.config.train_tensorboard and self.basedir is not None and
            not IS_TF_1 and not any(isinstance(cb,CARETensorBoardImage) for cb in self.callbacks)):
            self.callbacks.append(CARETensorBoardImage(model=self.keras_model, data=validation_data,
                                                       log_dir=str(self.logdir/'logs'/'images'),
                                                       n_images=3, prob_out=self.config.probabilistic))

        training_data = train.DataWrapper(X, Y, self.config.train_batch_size, length=epochs*steps_per_epoch)

        fit = self.keras_model.fit_generator if IS_TF_1 else self.keras_model.fit
        history = fit(iter(training_data), validation_data=validation_data,
                      epochs=epochs, steps_per_epoch=steps_per_epoch,
                      callbacks=self.callbacks, verbose=1)
        self._training_finished()

        return history


    @suppress_without_basedir(warn=True)
    def export_TF(self, fname=None):
        """Export neural network via :func:`csbdeep.utils.tf.export_SavedModel`.

        Parameters
        ----------
        fname : str or None
            Path of the created SavedModel archive (will end with ".zip").
            If ``None``, "<model-directory>/TF_SavedModel.zip" will be used.

        """
        if fname is None:
            fname = self.logdir / 'TF_SavedModel.zip'
        else:
            fname = Path(fname)

        meta = {
            'type':          self.__class__.__name__,
            'version':       package_version,
            'probabilistic': self.config.probabilistic,
            'axes':          self.config.axes,
            'axes_div_by':   self._axes_div_by(self.config.axes),
            'tile_overlap':  self._axes_tile_overlap(self.config.axes),
        }
        export_SavedModel(self.keras_model, str(fname), meta=meta)
        print("\nModel exported in TensorFlow's SavedModel format:\n%s" % str(fname.resolve()))


    def predict(self, img, axes, normalizer=PercentileNormalizer(), resizer=PadAndCropResizer(), n_tiles=None):
        """Apply neural network to raw image to predict restored image.

        Parameters
        ----------
        img : :class:`numpy.ndarray`
            Raw input image
        axes : str
            Axes of the input ``img``.
        normalizer : :class:`csbdeep.data.Normalizer` or None
            Normalization of input image before prediction and (potentially) transformation back after prediction.
        resizer : :class:`csbdeep.data.Resizer` or None
            If necessary, input image is resized to enable neural network prediction and result is (possibly)
            resized to yield original image size.
        n_tiles : iterable or None
            Out of memory (OOM) errors can occur if the input image is too large.
            To avoid this problem, the input image is broken up into (overlapping) tiles
            that can then be processed independently and re-assembled to yield the restored image.
            This parameter denotes a tuple of the number of tiles for every image axis.
            Note that if the number of tiles is too low, it is adaptively increased until
            OOM errors are avoided, albeit at the expense of runtime.
            A value of ``None`` denotes that no tiling should initially be used.

        Returns
        -------
        :class:`numpy.ndarray`
            Returns the restored image. If the model is probabilistic, this denotes the `mean` parameter of
            the predicted per-pixel Laplace distributions (i.e., the expected restored image).
            Axes semantics are the same as in the input image. Only if the output is multi-channel and
            the input image didn't have a channel axis, then output channels are appended at the end.

        """
        return self._predict_mean_and_scale(img, axes, normalizer, resizer, n_tiles)[0]


    def predict_probabilistic(self, img, axes, normalizer=PercentileNormalizer(), resizer=PadAndCropResizer(), n_tiles=None):
        """Apply neural network to raw image to predict probability distribution for restored image.

        See :func:`predict` for parameter explanations.

        Returns
        -------
        :class:`csbdeep.internals.probability.ProbabilisticPrediction`
            Returns the probability distribution of the restored image.

        Raises
        ------
        ValueError
            If this is not a probabilistic model.

        """
        self.config.probabilistic or _raise(ValueError('This is not a probabilistic model.'))
        mean, scale = self._predict_mean_and_scale(img, axes, normalizer, resizer, n_tiles)
        return ProbabilisticPrediction(mean, scale)


    def _predict_mean_and_scale(self, img, axes, normalizer, resizer, n_tiles=None):
        """Apply neural network to raw image to predict restored image.

        See :func:`predict` for parameter explanations.

        Returns
        -------
        tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray` or None)
            If model is probabilistic, returns a tuple `(mean, scale)` that defines the parameters
            of per-pixel Laplace distributions. Otherwise, returns the restored image via a tuple `(restored,None)`

        """
        normalizer, resizer = self._check_normalizer_resizer(normalizer, resizer)
        # axes = axes_check_and_normalize(axes,img.ndim)

        # different kinds of axes
        # -> typical case: net_axes_in = net_axes_out, img_axes_in = img_axes_out
        img_axes_in = axes_check_and_normalize(axes,img.ndim)
        net_axes_in = self.config.axes
        net_axes_out = axes_check_and_normalize(self._axes_out)
        set(net_axes_out).issubset(set(net_axes_in)) or _raise(ValueError("different kinds of output than input axes"))
        net_axes_lost = set(net_axes_in).difference(set(net_axes_out))
        img_axes_out = ''.join(a for a in img_axes_in if a not in net_axes_lost)
        # print(' -> '.join((img_axes_in, net_axes_in, net_axes_out, img_axes_out)))
        tiling_axes = net_axes_out.replace('C','') # axes eligible for tiling

        _permute_axes = self._make_permute_axes(img_axes_in, net_axes_in, net_axes_out, img_axes_out)
        # _permute_axes: (img_axes_in -> net_axes_in), undo: (net_axes_out -> img_axes_out)
        x = _permute_axes(img)
        # x has net_axes_in semantics
        x_tiling_axis = tuple(axes_dict(net_axes_in)[a] for a in tiling_axes) # numerical axis ids for x

        channel_in = axes_dict(net_axes_in)['C']
        channel_out = axes_dict(net_axes_out)['C']
        net_axes_in_div_by = self._axes_div_by(net_axes_in)
        net_axes_in_overlaps = self._axes_tile_overlap(net_axes_in)
        self.config.n_channel_in == x.shape[channel_in] or _raise(ValueError())

        # TODO: refactor tiling stuff to make code more readable

        def _total_n_tiles(n_tiles):
            n_block_overlaps = [int(np.ceil(1.* tile_overlap / block_size)) for tile_overlap, block_size in zip(net_axes_in_overlaps, net_axes_in_div_by)]
            return total_n_tiles(x,n_tiles=n_tiles,block_sizes=net_axes_in_div_by,n_block_overlaps=n_block_overlaps,guarantee='size')

        _permute_axes_n_tiles = self._make_permute_axes(img_axes_in, net_axes_in)
        # _permute_axes_n_tiles: (img_axes_in <-> net_axes_in) to convert n_tiles between img and net axes
        def _permute_n_tiles(n,undo=False):
            # hack: move tiling axis around in the same way as the image was permuted by creating an array
            return _permute_axes_n_tiles(np.empty(n,bool),undo=undo).shape

        # to support old api: set scalar n_tiles value for the largest tiling axis
        if np.isscalar(n_tiles) and int(n_tiles)==n_tiles and 1<=n_tiles:
            largest_tiling_axis = [i for i in np.argsort(x.shape) if i in x_tiling_axis][-1]
            _n_tiles = [n_tiles if i==largest_tiling_axis else 1 for i in range(x.ndim)]
            n_tiles = _permute_n_tiles(_n_tiles,undo=True)
            warnings.warn("n_tiles should be a tuple with an entry for each image axis")
            print("Changing n_tiles to %s" % str(n_tiles))

        if n_tiles is None:
            n_tiles = [1]*img.ndim
        try:
            n_tiles = tuple(n_tiles)
            img.ndim == len(n_tiles) or _raise(TypeError())
        except TypeError:
            raise ValueError("n_tiles must be an iterable of length %d" % img.ndim)

        all(np.isscalar(t) and 1<=t and int(t)==t for t in n_tiles) or _raise(
            ValueError("all values of n_tiles must be integer values >= 1"))
        n_tiles = tuple(map(int,n_tiles))
        n_tiles = _permute_n_tiles(n_tiles)
        (all(n_tiles[i] == 1 for i in range(x.ndim) if i not in x_tiling_axis) or
            _raise(ValueError("entry of n_tiles > 1 only allowed for axes '%s'" % tiling_axes)))
        # n_tiles_limited = self._limit_tiling(x.shape,n_tiles,net_axes_in_div_by)
        # if any(np.array(n_tiles) != np.array(n_tiles_limited)):
        #     print("Limiting n_tiles to %s" % str(_permute_n_tiles(n_tiles_limited,undo=True)))
        # n_tiles = n_tiles_limited
        n_tiles = list(n_tiles)


        # normalize & resize
        x = normalizer.before(x, net_axes_in)
        x = resizer.before(x, net_axes_in, net_axes_in_div_by)

        done = False
        progress = Progress(_total_n_tiles(n_tiles),1)
        c = 0
        while not done:
            try:
                # raise tf.errors.ResourceExhaustedError(None,None,None) # tmp
                x = predict_tiled(self.keras_model,x,axes_in=net_axes_in,axes_out=net_axes_out,
                                  n_tiles=n_tiles,block_sizes=net_axes_in_div_by,tile_overlaps=net_axes_in_overlaps,pbar=progress)
                # x has net_axes_out semantics
                done = True
                progress.close()
            except tf.errors.ResourceExhaustedError:
                # TODO: how to test this code?
                # n_tiles_prev = list(n_tiles) # make a copy
                tile_sizes_approx = np.array(x.shape) / np.array(n_tiles)
                t = [i for i in np.argsort(tile_sizes_approx) if i in x_tiling_axis][-1]
                n_tiles[t] *= 2
                # n_tiles = self._limit_tiling(x.shape,n_tiles,net_axes_in_div_by)
                # if all(np.array(n_tiles) == np.array(n_tiles_prev)):
                    # raise MemoryError("Tile limit exceeded. Memory occupied by another process (notebook)?")
                if c >= 8:
                    raise MemoryError("Giving up increasing number of tiles. Memory occupied by another process (notebook)?")
                print('Out of memory, retrying with n_tiles = %s' % str(_permute_n_tiles(n_tiles,undo=True)))
                progress.total = _total_n_tiles(n_tiles)
                c += 1

        n_channel_predicted = self.config.n_channel_out * (2 if self.config.probabilistic else 1)
        x.shape[channel_out] == n_channel_predicted or _raise(ValueError())

        x = resizer.after(x, net_axes_out)

        mean, scale = self._mean_and_scale_from_prediction(x,axis=channel_out)
        # mean and scale have net_axes_out semantics

        if normalizer.do_after and self.config.n_channel_in==self.config.n_channel_out:
            mean, scale = normalizer.after(mean, scale, net_axes_out)

        mean, scale = _permute_axes(mean,undo=True), _permute_axes(scale,undo=True)
        # mean and scale have img_axes_out semantics

        return mean, scale


    def _mean_and_scale_from_prediction(self,x,axis=-1):
        # separate mean and scale
        if self.config.probabilistic:
            _n = self.config.n_channel_out
            assert x.shape[axis] == 2*_n
            slices = [slice(None) for _ in x.shape]
            slices[axis] = slice(None,_n)
            mean = x[tuple(slices)]
            slices[axis] = slice(_n,None)
            scale = x[tuple(slices)]
        else:
            mean, scale = x, None
        return mean, scale

    # def _limit_tiling(self,img_shape,n_tiles,block_sizes):
    #     img_shape, n_tiles, block_sizes = np.array(img_shape), np.array(n_tiles), np.array(block_sizes)
    #     n_tiles_limit = np.ceil(img_shape / block_sizes) # each tile must be at least one block in size
    #     return [int(t) for t in np.minimum(n_tiles,n_tiles_limit)]

    def _axes_div_by(self, query_axes):
        query_axes = axes_check_and_normalize(query_axes)
        # default: must be divisible by power of 2 to allow down/up-sampling steps in unet
        pool_div_by = 2**self.config.unet_n_depth
        return tuple((pool_div_by if a in 'XYZT' else 1) for a in query_axes)

    def _axes_tile_overlap(self, query_axes):
        query_axes = axes_check_and_normalize(query_axes)
        overlap = tile_overlap(self.config.unet_n_depth, self.config.unet_kern_size)
        return tuple((overlap if a in 'XYZT' else 0) for a in query_axes)

    @property
    def _config_class(self):
        return Config

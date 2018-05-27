from __future__ import print_function, unicode_literals, absolute_import, division

import datetime
import warnings

import tensorflow as tf
from six import string_types

from csbdeep.internals.probability import ProbabilisticPrediction
from .config import Config

from ..utils import _raise, Path, load_json, save_json, axes_check_and_normalize, axes_dict, move_image_axes
from ..utils.tf import export_SavedModel
from ..version import __version__ as package_version
from ..data import Normalizer, NoNormalizer, PercentileNormalizer
from ..data import Resizer, NoResizer, PadAndCropResizer
from ..internals.predict import predict_direct, predict_tiled, tile_overlap
from ..internals import nets, train


class CARE(object):
    """Standard CARE network for image restoration and enhancement.

    Uses a convolutional neural network created by :func:`csbdeep.nets.common_unet`.
    Note that isotropic reconstruction and manifold extraction/projection are not supported here
    (see :class:`csbdeep.models.IsotropicCARE`).

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
        Model's name.
    logdir : :class:`pathlib.Path`
        Path to model's folder (which stores configuration, weights, etc.)
    """

    def __init__(self, config, name=None, basedir='.'):
        """See class docstring."""

        config is None or isinstance(config,Config) or _raise(ValueError('Invalid configuration: %s' % str(config)))
        if config is not None and not config.is_valid():
            invalid_attr = config.is_valid(True)[1]
            raise ValueError('Invalid configuration attributes: ' + ', '.join(invalid_attr))

        name is None or isinstance(name,string_types) or _raise(ValueError())
        isinstance(basedir,(string_types,Path)) or _raise(ValueError())
        self.config = config
        self.basedir = Path(basedir)
        self.name = name
        self._set_logdir()
        self._model_prepared = False
        self.keras_model = self._build()
        if config is None:
            self._find_and_load_weights()


    def _set_logdir(self):
        if self.name is None:
            self.name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
        self.logdir = self.basedir / self.name

        config_file =  self.logdir / 'config.json'
        if self.config is None:
            if config_file.exists():
                config_dict = load_json(str(config_file))
                self.config = Config(**config_dict)
                if not self.config.is_valid():
                    invalid_attr = self.config.is_valid(True)[1]
                    raise ValueError('Invalid attributes in loaded config: ' + ', '.join(invalid_attr))
            else:
                raise FileNotFoundError("config file doesn't exist: %s" % str(config_file.resolve()))
        else:
            if self.logdir.exists():
                warnings.warn('output path for model already exists, files may be overwritten: %s' % str(self.logdir.resolve()))
            self.logdir.mkdir(parents=True, exist_ok=True)
            save_json(vars(self.config), str(config_file))


    def _find_and_load_weights(self,prefer='best'):
        import sys
        from itertools import chain
        # get all weight files and sort by modification time descending (newest first)
        weights_ext   = ('*.h5','*.hdf5')
        weights_files = chain(*(self.logdir.glob(ext) for ext in weights_ext))
        weights_files = reversed(sorted(weights_files, key=lambda f: f.lstat().st_mtime))
        weights_files = list(weights_files)
        if len(weights_files) == 0:
            print("Couldn't find any network weights (%s) to load." % ', '.join(weights_ext), file=sys.stderr)
            return
        weights_preferred = list(filter(lambda f: prefer in f.name, weights_files))
        weights_chosen = weights_preferred[0] if len(weights_preferred)>0 else weights_files[0]
        print("Loading network weights from '%s'." % weights_chosen.name)
        self.load_weights(weights_chosen.name)


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


    def load_weights(self, name='weights_best.h5'):
        """Load neural network weights from model folder.

        Parameters
        ----------
        name : str
            Name of HDF5 weight file (as saved during or after training).
        """
        self.keras_model.load_weights(str(self.logdir/name))


    def prepare_for_training(self, optimizer=None, **kwargs):
        """Prepare for neural network training.

        Calls :func:`csbdeep.train.prepare_model` and creates
        `Keras Callbacks <https://keras.io/callbacks/>`_ to be used for training.

        Note that this method will be implicitly called once by :func:`train`
        (with default arguments) if not done so explicitly beforehand.

        Parameters
        ----------
        optimizer : obj or None
            Instance of a `Keras Optimizer <https://keras.io/optimizers/>`_ to be used for training.
            If ``None`` (default), uses ``Adam`` with the learning rate specified in ``config``.
        kwargs : dict
            Additional arguments for :func:`csbdeep.train.prepare_model`.

        """
        if optimizer is None:
            from keras.optimizers import Adam
            optimizer = Adam(lr=self.config.train_learning_rate)
        self.callbacks = train.prepare_model(self.keras_model, optimizer, self.config.train_loss, **kwargs)

        if self.config.train_checkpoint is not None:
            from keras.callbacks import ModelCheckpoint
            self.callbacks.append(ModelCheckpoint(str(self.logdir / self.config.train_checkpoint), save_best_only=True, save_weights_only=True))

        if self.config.train_tensorboard:
            from ..utils.tf import CARETensorBoard
            self.callbacks.append(CARETensorBoard(log_dir=str(self.logdir), prefix_with_timestamp=False, n_images=3, write_images=True, prob_out=self.config.probabilistic))

        if self.config.train_reduce_lr is not None:
            from keras.callbacks import ReduceLROnPlateau
            rlrop_params = self.config.train_reduce_lr
            if 'verbose' not in rlrop_params:
                rlrop_params['verbose'] = True
            self.callbacks.append(ReduceLROnPlateau(**rlrop_params))

        self._model_prepared = True


    def train(self, X,Y, validation_data, epochs=None, steps_per_epoch=None):
        """Train the neural network with the given data.

        Parameters
        ----------
        X : :class:`numpy.ndarray`
            Array of source images
        Y : :class:`numpy.ndarray`
            Array of target images
        validation_data : tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`)
            Tuple of arrays for source and target validation images
        epochs : int
            Optional argument to use instead of the value from ``config``.
        steps_per_epoch : int
            Optional argument to use instead of the value from ``config``.

        Returns
        -------
        ``History`` object
            See `Keras training history <https://keras.io/models/model/#fit>`_.

        """
        if not self._model_prepared:
            self.prepare_for_training()

        # if self.logdir.exists():
        #     warnings.warn('output path for model already exists, files may be overwritten during training: %s' % str(self.logdir.resolve()))

        axes = axes_check_and_normalize('S'+self.config.axes,X.ndim)
        ax = axes_dict(axes)
        div_by = 2**self.config.unet_n_depth
        axes_relevant = ''.join(a for a in 'XYZT' if a in axes)
        for a in axes_relevant:
            n = X.shape[ax[a]]
            if n % div_by != 0:
                raise ValueError(
                    "training images must be evenly divisible by %d along axes %s"
                    " (axis %s has incompatible size %d)" % (div_by,axes_relevant,a,n)
                )

        if epochs is None:
            epochs = self.config.train_epochs
        if steps_per_epoch is None:
            steps_per_epoch = self.config.train_steps_per_epoch

        training_data = train.DataWrapper(X, Y, self.config.train_batch_size)

        history = self.keras_model.fit_generator(generator=training_data, validation_data=validation_data,
                                                 epochs=epochs, steps_per_epoch=steps_per_epoch,
                                                 callbacks=self.callbacks, verbose=1)

        self.keras_model.save_weights(str(self.logdir/'weights_final.h5'))
        return history


    def export_TF(self):
        """Export neural network via :func:`csbdeep.tf.export_SavedModel`."""
        fout = self.logdir / 'TF_SavedModel.zip'
        meta = {
            'version':       package_version,
            'probabilistic': self.config.probabilistic,
            'axes':          self.config.axes,
            'axes_div_by':   [(2**self.config.unet_n_depth if a in 'XYZT' else 1) for a in self.config.axes],
            'tile_overlap':  tile_overlap(self.config.unet_n_depth, self.config.unet_kern_size),
        }
        export_SavedModel(self.keras_model, str(fout), meta=meta)
        print("\nModel exported in TensorFlow's SavedModel format:\n%s" % str(fout.resolve()))


    def predict(self, img, axes, normalizer=PercentileNormalizer(), resizer=PadAndCropResizer(), n_tiles=1):
        """Apply neural network to raw image to predict restored image.

        Parameters
        ----------
        img : :class:`numpy.ndarray`
            Raw input image, with image dimensions expected in the same order as in data for training.
            If applicable, only the channel dimension can be anywhere.
            TODO: docstrings need update now that "axes" is used.
        axes : str
            Axes of ``img``.
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
        return self._predict_mean_and_scale(img, axes, normalizer, resizer, n_tiles)[0]


    def predict_probabilistic(self, img, axes, normalizer=PercentileNormalizer(), resizer=PadAndCropResizer(), n_tiles=1):
        """Apply neural network to raw image to predict probability distribution for restored image.

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
        mean, scale = self._predict_mean_and_scale(img, axes, normalizer, resizer, n_tiles)
        return ProbabilisticPrediction(mean, scale)


    def _predict_mean_and_scale(self, img, axes, normalizer, resizer, n_tiles=1):
        """Apply neural network to raw image to predict restored image.

        See :func:`predict` for parameter explanations.

        Returns
        -------
        tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray` or None)
            If model is probabilistic, returns a tuple `(mean, scale)` that defines the parameters
            of per-pixel Laplace distributions. Otherwise, returns the restored image via a tuple `(restored,None)`

        """
        normalizer, resizer = self._check_normalizer_resizer(normalizer, resizer)
        axes = axes_check_and_normalize(axes,img.ndim)
        _permute_axes = self._make_permute_axes(axes, self.config.axes)

        x = _permute_axes(img)
        channel = axes_dict(self.config.axes)['C']

        self.config.n_channel_in == x.shape[channel] or _raise(ValueError())

        # normalize
        x = normalizer.before(x,self.config.axes)
        # resize: make divisible by power of 2 to allow downsampling steps in unet
        div_n = 2 ** self.config.unet_n_depth
        x = resizer.before(x,div_n,exclude=channel)

        done = False
        while not done:
            try:
                if n_tiles == 1:
                    x = predict_direct(self.keras_model,x,channel_in=channel,channel_out=channel)
                else:
                    overlap = tile_overlap(self.config.unet_n_depth, self.config.unet_kern_size)
                    x = predict_tiled(self.keras_model,x,channel_in=channel,channel_out=channel,
                                      n_tiles=n_tiles,block_size=div_n,tile_overlap=overlap)
                done = True
            except tf.errors.ResourceExhaustedError:
                n_tiles = max(4, 2*n_tiles)
                print('Out of memory, retrying with n_tiles = %d' % n_tiles)

        n_channel_predicted = self.config.n_channel_out * (2 if self.config.probabilistic else 1)
        x.shape[channel] == n_channel_predicted or _raise(ValueError())

        x = resizer.after(x,exclude=channel)

        mean, scale = self._mean_and_scale_from_prediction(x,axis=channel)

        if normalizer.do_after and self.config.n_channel_in==self.config.n_channel_out:
            mean, scale = normalizer.after(mean, scale)

        mean, scale = _permute_axes(mean,undo=True), _permute_axes(scale,undo=True)

        return mean, scale


    def _mean_and_scale_from_prediction(self,x,axis=-1):
        # separate mean and scale
        if self.config.probabilistic:
            _n = self.config.n_channel_out
            assert x.shape[axis] == 2*_n
            slices = [slice(None) for _ in x.shape]
            slices[axis] = slice(None,_n)
            mean = x[slices]
            slices[axis] = slice(_n,None)
            scale = x[slices]
        else:
            mean, scale = x, None
        return mean, scale

    def _make_permute_axes(self,axes_in,axes_out=None):
        if axes_out is None:
            axes_out = self.config.axes
        channel_in  = axes_dict(axes_in) ['C']
        channel_out = axes_dict(axes_out)['C']
        assert channel_out is not None

        def _permute_axes(data,undo=False):
            if data is None:
                return None
            if undo:
                if channel_in is not None:
                    return move_image_axes(data, axes_out, axes_in, True)
                else:
                    # input is single-channel and has no channel axis
                    data = move_image_axes(data, axes_out, axes_in+'C', True)
                    # output is single-channel -> remove channel axis
                    if data.shape[-1] == 1:
                        data = data[...,0]
                    return data
            else:
                return move_image_axes(data, axes_in, axes_out, True)
        return _permute_axes

    def _check_normalizer_resizer(self, normalizer, resizer):
        if normalizer is None:
            normalizer = NoNormalizer()
        if resizer is None:
            resizer = NoResizer()
        isinstance(resizer,Resizer) or _raise(ValueError())
        isinstance(normalizer,Normalizer) or _raise(ValueError())
        if normalizer.do_after:
            if self.config.n_channel_in != self.config.n_channel_out:
                warnings.warn('skipping normalization step after prediction because ' +
                              'number of input and output channels differ.')

        return normalizer, resizer

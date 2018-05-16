from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter
from six import string_types

import argparse
import datetime

from .utils import _raise, consume, Path, load_json, save_json, is_tf_dim, rotate, axes_check_and_normalize, axes_dict, move_image_axes
import warnings
import numpy as np
# from collections import namedtuple
from keras import backend as K
import tensorflow as tf

from . import nets, train
from .predict import predict_direct, predict_tiled, tile_overlap, Normalizer, Resizer, PadAndCropResizer
from .probability import ProbabilisticPrediction


class Config(argparse.Namespace):
    """Default configuration for a (standard) CARE network.

    This configuration is meant to be used with :class:`CARE`
    and related models (e.g., :class:`IsotropicCARE`).

    Parameters
    ----------
    axes : str
        Axes of the neural network (channel axis optional).
    n_channel_in : int
        Number of channels of given input image.
    n_channel_out : int
        Number of channels of predicted output image.
    probabilistic : bool
        Probabilistic prediction of per-pixel Laplace distributions or
        typical regression of per-pixel scalar values.
    kwargs : dict
        Overwrite (or add) configuration attributes (see below).

    Example
    -------
    >>> config = Config('YX', probabilistic=True, unet_n_depth=3)

    Attributes
    ----------
    n_dim : int
        Dimensionality of input images (2 or 3).
    unet_residual : bool
        Parameter `residual` of :func:`csbdeep.nets.common_unet`. Default: ``n_channel_in == n_channel_out``
    unet_n_depth : int
        Parameter `n_depth` of :func:`csbdeep.nets.common_unet`. Default: ``2``
    unet_kern_size : int
        Parameter `kern_size` of :func:`csbdeep.nets.common_unet`. Default: ``5 if n_dim==2 else 3``
    unet_n_first : int
        Parameter `n_first` of :func:`csbdeep.nets.common_unet`. Default: ``32``
    unet_last_activation : str
        Parameter `last_activation` of :func:`csbdeep.nets.common_unet`. Default: ``linear``
    train_loss : str
        Name of training loss. Default: ``'laplace' if probabilistic else 'mae'``
    train_epochs : int
        Number of training epochs. Default: ``100``
    train_steps_per_epoch : int
        Number of parameter update steps per epoch. Default: ``400``
    train_learning_rate : float
        Learning rate for training. Default: ``0.0004``
    train_batch_size : int
        Batch size for training. Default: ``16``
    train_tensorboard : bool
        Enable TensorBoard for monitoring training progress. Default: ``True``
    train_checkpoint : str
        Name of checkpoint file for model weights (only best are saved); set to ``None`` to disable. Default: ``weights_best.h5``
    train_reduce_lr : dict
        Parameter :class:`dict` of ReduceLROnPlateau_ callback; set to ``None`` to disable. Default: ``{'factor': 0.5, 'patience': 10}``

        .. _ReduceLROnPlateau: https://keras.io/callbacks/#reducelronplateau
    """

    def __init__(self, axes, n_channel_in=1, n_channel_out=1, probabilistic=False, **kwargs):
        """See class docstring."""

        # parse and check axes
        axes = axes_check_and_normalize(axes)
        ax = axes_dict(axes)
        ax = {a: (ax[a] is not None) for a in ax}

        (ax['X'] and ax['Y']) or _raise(ValueError('lateral axes X and Y must be present.'))
        not (ax['Z'] and ax['T']) or _raise(ValueError('using Z and T axes together not supported.'))

        axes.startswith('S') or (not ax['S']) or _raise(ValueError('sample axis S must be first.'))
        axes = axes.replace('S','') # remove sample axis if it exists

        n_dim = 3 if (ax['Z'] or ax['T']) else 2

        # TODO: Config not independent of backend. Problem?
        # could move things around during train/predict as an alternative... good idea?
        # otherwise, users can choose axes of input image anyhow, so doesn't matter if model is fixed to something else
        assert K.image_data_format() in ('channels_first','channels_last')
        if K.image_data_format() == 'channels_last':
            if ax['C']:
                axes[-1] == 'C' or _raise(ValueError('channel axis must be last for backend (%s).' % K.backend()))
            else:
                axes += 'C'
        else:
            if ax['C']:
                axes[0] == 'C' or _raise(ValueError('channel axis must be first for backend (%s).' % K.backend()))
            else:
                axes = 'C'+axes

        # directly set by parameters
        self.n_dim                 = n_dim
        self.axes                  = axes
        self.n_channel_in          = int(n_channel_in)
        self.n_channel_out         = int(n_channel_out)
        self.probabilistic         = bool(probabilistic)

        # default config (can be overwritten by kwargs below)
        self.unet_residual         = self.n_channel_in == self.n_channel_out
        self.unet_n_depth          = 2
        self.unet_kern_size        = 5 if self.n_dim==2 else 3
        self.unet_n_first          = 32
        self.unet_last_activation  = 'linear'
        self.unet_input_shape      = self.n_dim*(None,) + (self.n_channel_in,)

        self.train_loss            = 'laplace' if self.probabilistic else 'mae'
        self.train_epochs          = 100
        self.train_steps_per_epoch = 400
        self.train_learning_rate   = 0.0004
        self.train_batch_size      = 16
        self.train_tensorboard     = True
        self.train_checkpoint      = 'weights_best.h5'
        self.train_reduce_lr       = {'factor': 0.5, 'patience': 10}

        # disallow setting 'n_dim' manually
        try:
            del kwargs['n_dim']
            # warnings.warn("ignoring parameter 'n_dim'")
        except:
            pass

        for k in kwargs:
            setattr(self, k, kwargs[k])


    def is_valid(self, return_invalid=False):
        """Check if configuration is valid.

        Returns
        -------
        bool
            Flag that indicates whether the current configuration values are valid.
        """
        def _is_int(v,low=None,high=None):
            return (
                isinstance(v,int) and
                (True if low is None else low <= v) and
                (True if high is None else v <= high)
            )

        ok = {}
        ok['n_dim'] = self.n_dim in (2,3)
        try:
            axes_check_and_normalize(self.axes,self.n_dim+1,disallowed='S')
            ok['axes'] = True
        except:
            ok['axes'] = False
        ok['n_channel_in']  = _is_int(self.n_channel_in,1)
        ok['n_channel_out'] = _is_int(self.n_channel_out,1)
        ok['probabilistic'] = isinstance(self.probabilistic,bool)

        ok['unet_residual'] = (
            isinstance(self.unet_residual,bool) and
            (not self.unet_residual or (self.n_channel_in==self.n_channel_out))
        )
        ok['unet_n_depth']         = _is_int(self.unet_n_depth,1)
        ok['unet_kern_size']       = _is_int(self.unet_kern_size,1)
        ok['unet_n_first']         = _is_int(self.unet_n_first,1)
        ok['unet_last_activation'] = self.unet_last_activation in ('linear','relu')
        ok['unet_input_shape'] = (
            isinstance(self.unet_input_shape,(list,tuple)) and
            len(self.unet_input_shape) == self.n_dim+1 and
            self.unet_input_shape[-1] == self.n_channel_in and
            all((d is None or (_is_int(d) and d%(2**self.unet_n_depth)==0) for d in self.unet_input_shape[:-1]))
        )
        ok['train_loss'] = (
            (    self.probabilistic and self.train_loss == 'laplace'   ) or
            (not self.probabilistic and self.train_loss in ('mse','mae'))
        )
        ok['train_epochs']          = _is_int(self.train_epochs,1)
        ok['train_steps_per_epoch'] = _is_int(self.train_steps_per_epoch,1)
        ok['train_learning_rate']   = np.isscalar(self.train_learning_rate) and self.train_learning_rate > 0
        ok['train_batch_size']      = _is_int(self.train_batch_size,1)
        ok['train_tensorboard']     = isinstance(self.train_tensorboard,bool)
        ok['train_checkpoint']      = self.train_checkpoint is None or isinstance(self.train_checkpoint,string_types)
        ok['train_reduce_lr']       = self.train_reduce_lr  is None or isinstance(self.train_reduce_lr,dict)

        if return_invalid:
            return all(ok.values()), tuple(k for (k,v) in ok.items() if not v)
        else:
            return all(ok.values())



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
            from csbdeep.tf import CARETensorBoard
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
        from csbdeep.tf import export_SavedModel
        fout = self.logdir / 'TF_SavedModel.zip'
        meta = {
            'axes':         self.config.axes,
            'div_by':       2**self.config.unet_n_depth,
            'tile_overlap': tile_overlap(self.config.unet_n_depth, self.config.unet_kern_size),
        }
        export_SavedModel(self.keras_model, str(fout), meta=meta)
        print("\nModel exported in TensorFlow's SavedModel format:\n%s" % str(fout.resolve()))


    def predict(self, img, axes, normalizer, resizer=PadAndCropResizer(), n_tiles=1):
        """Apply neural network to raw image to predict restored image.

        Parameters
        ----------
        img : :class:`numpy.ndarray`
            Raw input image, with image dimensions expected in the same order as in data for training.
            If applicable, only the channel dimension can be anywhere.
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


    def predict_probabilistic(self, img, axes, normalizer, resizer=PadAndCropResizer(), n_tiles=1):
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
        axes = axes_check_and_normalize(axes,img.ndim)
        _permute_axes = self._make_permute_axes(axes, self.config.axes)

        x = _permute_axes(img)
        channel = axes_dict(self.config.axes)['C']

        # print(x.shape, channel)
        self.config.n_channel_in == x.shape[channel] or _raise(ValueError())
        isinstance(resizer,Resizer) or _raise(ValueError())
        isinstance(normalizer,Normalizer) or _raise(ValueError())

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

        if normalizer.do_after:
            self.config.n_channel_in == self.config.n_channel_out or _raise(ValueError())
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



class IsotropicCARE(CARE):
    """CARE network for isotropic image reconstruction.

    Extends :class:`csbdeep.models.CARE` by replacing prediction
    (:func:`predict`, :func:`predict_probabilistic`) to do isotropic reconstruction.
    """

    def predict(self, img, axes, factor, normalizer, resizer=PadAndCropResizer(), batch_size=8):
        """Apply neural network to raw image to restore isotropic resolution.

        Parameters
        ----------
        img : :class:`numpy.ndarray`
            Raw input image, with image dimensions expected in the same order as in data for training.
            If applicable, only the z and channel dimensions can be anywhere.
        axes : str
            Axes of ``img``.
        factor : int
            Upsampling factor for z dimension. It is important that this is chosen in correspondence
            to the subsampling factor used during training data generation.
        normalizer : :class:`csbdeep.predict.Normalizer`
            Normalization of input image before prediction and (potentially) transformation back after prediction.
        resizer : :class:`csbdeep.predict.Resizer`
            If necessary, input image is resized to enable neural network prediction and result is (possibly)
            resized to yield original image size.
        batch_size : int
            Number of image slices that are processed together by the neural network.
            Reduce this value if out of memory errors occur.

        Returns
        -------
        :class:`numpy.ndarray`
            Returns the restored image. If the model is probabilistic, this denotes the `mean` parameter of
            the predicted per-pixel Laplace distributions (i.e., the expected restored image).
            Axes ordering is unchanged wrt input image. Only if there the output is multi-channel and
            the input image didn't have a channel axis, then channels are appended at the end.

        """
        return self._predict_mean_and_scale(img, axes, factor, normalizer, resizer, batch_size)[0]


    def predict_probabilistic(self, img, axes, factor, normalizer, resizer=PadAndCropResizer(), batch_size=8):
        """Apply neural network to raw image to predict probability distribution for isotropic restored image.

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
        mean, scale = self._predict_mean_and_scale(img, axes, factor, normalizer, resizer, batch_size)
        return ProbabilisticPrediction(mean, scale)


    def _predict_mean_and_scale(self, img, axes, factor, normalizer, resizer, batch_size=8):
        """Apply neural network to raw image to restore isotropic resolution.

        See :func:`predict` for parameter explanations.

        Returns
        -------
        tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray` or None)
            If model is probabilistic, returns a tuple `(mean, scale)` that defines the parameters
            of per-pixel Laplace distributions. Otherwise, returns the restored image via a tuple `(restored,None)`

        Todo
        ----
        - :func:`scipy.ndimage.interpolation.zoom` differs from :func:`gputools.scale`. Important?

        """
        axes = axes_check_and_normalize(axes,img.ndim)
        'Z' in axes or _raise(ValueError())
        axes_tmp = 'CZ' + axes.replace('Z','').replace('C','')
        _permute_axes = self._make_permute_axes(axes, axes_tmp)
        channel = 0

        x = _permute_axes(img)

        # print(x.shape, channel)
        self.config.n_channel_in == x.shape[channel] or _raise(ValueError())
        np.isscalar(factor) and factor > 0 or _raise(ValueError())
        isinstance(resizer,Resizer) or _raise(ValueError())
        isinstance(normalizer,Normalizer) or _raise(ValueError())

        from scipy.ndimage.interpolation import zoom
        def scale_z(arr,factor):
            return zoom(arr,(1,factor,1,1),order=1)

        # scale z up (second axis)
        x_scaled = scale_z(x,factor)

        # normalize
        x = normalizer.before(x,axes_tmp)
        # resize: make (x,y,z) image dimensions divisible by power of 2 to allow downsampling steps in unet
        div_n = 2 ** self.config.unet_n_depth
        x_scaled = resizer.before(x_scaled,div_n,exclude=channel)

        # move channel to the end
        x_scaled = np.moveaxis(x_scaled, channel, -1)
        channel = -1

        # u1: first rotation and prediction
        x_rot1   = rotate(x_scaled, axis=1, copy=False)
        u_rot1   = predict_direct(self.keras_model, x_rot1, channel_in=channel, channel_out=channel, single_sample=False,
                                  batch_size=batch_size, verbose=0)
        u1       = rotate(u_rot1, -1, axis=1, copy=False)

        # u2: second rotation and prediction
        x_rot2   = rotate(rotate(x_scaled, axis=2, copy=False), axis=0, copy=False)
        u_rot2   = predict_direct(self.keras_model, x_rot2, channel_in=channel, channel_out=channel, single_sample=False,
                                  batch_size=batch_size, verbose=0)
        u2       = rotate(rotate(u_rot2, -1, axis=0, copy=False), -1, axis=2, copy=False)

        n_channel_predicted = self.config.n_channel_out * (2 if self.config.probabilistic else 1)
        u_rot1.shape[channel] == n_channel_predicted or _raise(ValueError())
        u_rot2.shape[channel] == n_channel_predicted or _raise(ValueError())

        # move channel back to the front
        u1 = np.moveaxis(u1, channel, 0)
        u2 = np.moveaxis(u2, channel, 0)
        channel = 0

        # resize after prediction
        u1 = resizer.after(u1,exclude=channel)
        u2 = resizer.after(u2,exclude=channel)

        # combine u1 & u2
        mean1, scale1 = self._mean_and_scale_from_prediction(u1,axis=channel)
        mean2, scale2 = self._mean_and_scale_from_prediction(u2,axis=channel)
        # avg = lambda u1,u2: (u1+u2)/2 # arithmetic mean
        avg = lambda u1,u2: np.sqrt(np.maximum(u1,0)*np.maximum(u2,0)) # geometric mean
        mean, scale = avg(mean1,mean2), None
        if self.config.probabilistic:
            scale = np.maximum(scale1,scale2)

        if normalizer.do_after:
            self.config.n_channel_in == self.config.n_channel_out or _raise(ValueError())
            mean, scale = normalizer.after(mean, scale)

        mean, scale = _permute_axes(mean,undo=True), _permute_axes(scale,undo=True)

        return mean, scale

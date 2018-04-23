from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

import argparse
import datetime

from .utils import _raise, consume, Path, load_json, save_json, is_tf_dim, rotate
import warnings
import numpy as np
# from collections import namedtuple
import tensorflow as tf

from . import nets, train
from .predict import predict_direct, predict_tiled, tile_overlap, Normalizer, Resizer, PadAndCropResizer


class Config(argparse.Namespace):
    """Default configuration for a (standard) CARE network.

    This configuration is meant to be used with :class:`csbdeep.models.CARE`.

    Parameters
    ----------
    n_dim : int
        Dimensionality of input images (2 or 3).
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
    >>> config = Config(2, probabilistic=True, unet_n_depth=3)

    Attributes
    ----------
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

    def __init__(self, n_dim, n_channel_in=1, n_channel_out=1, probabilistic=False, **kwargs):
        """See class docstring."""
        n_dim in (2,3) or _raise(ValueError())

        self.n_dim                 = n_dim
        self.n_channel_in          = n_channel_in
        self.n_channel_out         = n_channel_out
        self.probabilistic         = probabilistic

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

        for k in kwargs:
            setattr(self, k, kwargs[k])


    def is_valid(self):
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
        # TODO: make nicer and terminate early on False
        _v  = True
        _v &= self.n_dim in (2,3)
        _v &= _is_int(self.n_channel_in,1)
        _v &= _is_int(self.n_channel_out,1)
        _v &= isinstance(self.probabilistic,bool)
        _v &= (isinstance(self.unet_residual,bool) and
               (not self.unet_residual or (self.n_channel_in==self.n_channel_out)))
        _v &= _is_int(self.unet_n_depth,1)
        _v &= _is_int(self.unet_kern_size,1)
        _v &= _is_int(self.unet_n_first,1)
        _v &= self.unet_last_activation in ('linear','relu')
        _v &= (isinstance(self.unet_input_shape,tuple) and
               (len(self.unet_input_shape)==self.n_dim+1) and
               (self.unet_input_shape[-1]==self.n_channel_in) and
               all((d is None or (_is_int(d) and d%(2**self.unet_n_depth)==0) for d in self.unet_input_shape[:-1])))
        _v &= ((    self.probabilistic and self.train_loss == 'laplace'   ) or
               (not self.probabilistic and self.train_loss in ('mse','mae')))
        _v &= _is_int(self.train_epochs,1)
        _v &= _is_int(self.train_steps_per_epoch,1)
        _v &= np.isscalar(self.train_learning_rate) and self.train_learning_rate > 0
        _v &= _is_int(self.train_batch_size,1)
        _v &= isinstance(self.train_tensorboard,bool)
        _v &= self.train_checkpoint is None or isinstance(self.train_checkpoint,str)
        _v &= self.train_reduce_lr  is None or isinstance(self.train_reduce_lr,dict)
        return _v



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
    outdir : str
        Output directory that contains (or will contain) a folder with the given model name.

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

    def __init__(self, config, name=None, outdir='.'):
        """See class docstring."""
        (config is None or (isinstance(config,Config) and config.is_valid())
            or _raise(ValueError('Invalid configuration: %s' % str(config))))
        name is None or isinstance(name,str) or _raise(ValueError())
        isinstance(outdir,(str,Path)) or _raise(ValueError())
        self.config = config
        self.outdir = Path(outdir)
        self.name = name
        self._set_logdir()
        self._model_prepared = False
        self.keras_model = self._build()


    def _set_logdir(self):
        if self.name is None:
            self.name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
        self.logdir = self.outdir / self.name

        config_file =  self.logdir / 'config.json'
        if self.config is None:
            if not config_file.exists():
                raise FileNotFoundError("config file doesn't exist: %s" % str(config_file.resolve()))
            else:
                config_dict = load_json(config_file)
                self.config = Config(**config_dict)
        else:
            if self.logdir.exists():
                warnings.warn('output path for model already exists, files may be overwritten: %s' % str(self.logdir.resolve()))
            self.logdir.mkdir(parents=True, exist_ok=True)
            save_json(vars(self.config), config_file)


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
            self.callbacks.append(ReduceLROnPlateau(**self.config.train_reduce_lr, verbose=True))

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
        export_SavedModel(self.keras_model, str(fout))
        print("\nModel exported in TensorFlow's SavedModel format:\n%s" % str(fout.resolve()))


    def predict(self, img, normalizer, resizer=PadAndCropResizer(), channel=None, n_tiles=1):
        """Apply neural network to raw image to predict restored image.

        Parameters
        ----------
        img : :class:`numpy.ndarray`
            Raw input image, with image dimensions expected in the same order as in data for training.
            If applicable, only the channel dimension can be anywhere.
        normalizer : :class:`csbdeep.predict.Normalizer`
            Normalization of input image before prediction and (potentially) transformation back after prediction.
        resizer : :class:`csbdeep.predict.Resizer`
            If necessary, input image is resized to enable neural network prediction and result is (possibly)
            resized to yield original image size.
        channel : int or None
            Index of channel dimension of raw input image. Defaults to ``None``, assuming
            a single-channel input image where without an explicit channel dimension.
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

        """
        return self._predict_mean_and_scale(img, normalizer, resizer, channel, n_tiles)[0]


    def _predict_mean_and_scale(self, img, normalizer, resizer, channel=None, n_tiles=1):
        """Apply neural network to raw image to predict restored image.

        Parameters
        ----------
        img : :class:`numpy.ndarray`
            Raw input image, with image dimensions expected in the same order as in data for training.
            If applicable, only the channel dimension can be anywhere.
        normalizer : :class:`csbdeep.predict.Normalizer`
            Normalization of input image before prediction and (potentially) transformation back after prediction.
        resizer : :class:`csbdeep.predict.Resizer`
            If necessary, input image is resized to enable neural network prediction and result is (possibly)
            resized to yield original image size.
        channel : int or None
            Index of channel dimension of raw input image. Defaults to ``None``, assuming
            a single-channel input image where without an explicit channel dimension.
        n_tiles : int
            Out of memory (OOM) errors can occur if the input image is too large.
            To avoid this problem, the input image is broken up into (overlapping) tiles
            that can then be processed independently and re-assembled to yield the restored image.
            This parameter denotes the number of tiles. Note that if the number of tiles is too low,
            it is adaptively increased until OOM errors are avoided, albeit at the expense of runtime.

        Returns
        -------
        tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray` or None)
            If model is probabilistic, returns a tuple `(mean, scale)` that defines the parameters
            of per-pixel Laplace distributions. Otherwise, returns the restored image via a tuple `(restored,None)`

        """
        if channel is None:
            self.config.n_channel_in == 1 or _raise(ValueError())
        else:
            -img.ndim <= channel < img.ndim or _raise(ValueError())
            if channel < 0:
                channel %= img.ndim
            self.config.n_channel_in == img.shape[channel] or _raise(ValueError())

        n_channel_predicted = self.config.n_channel_out * (2 if self.config.probabilistic else 1)

        isinstance(resizer,Resizer) or _raise(ValueError())
        isinstance(normalizer,Normalizer) or _raise(ValueError())

        # normalize
        x = normalizer.before(img,channel)

        # resize: make divisible by power of 2 to allow downsampling steps in unet
        div_n = 2 ** self.config.unet_n_depth
        x = resizer.before(x,div_n,channel)

        done = False
        while not done:
            try:
                if n_tiles == 1:
                    x = predict_direct(self.keras_model,x,channel_in=channel,channel_out=0)
                else:
                    overlap = tile_overlap(self.config.unet_n_depth, self.config.unet_kern_size)
                    x = predict_tiled(self.keras_model,x,channel_in=channel,channel_out=0,
                                      n_tiles=n_tiles,block_size=div_n,tile_overlap=overlap)
                done = True
            except tf.errors.ResourceExhaustedError:
                n_tiles = max(4, 2*n_tiles)
                print('Out of memory, retrying with n_tiles = %d' % n_tiles)

        x.shape[0] == n_channel_predicted or _raise(ValueError())

        x = resizer.after(x,exclude=0)

        # separate mean and scale
        if self.config.probabilistic:
            _n = self.config.n_channel_out
            mean, scale = x[:_n], x[_n:]
        else:
            mean, scale = x, None

        if channel is not None:
            # move output channel to same dimension as in input image
            mean = np.moveaxis(mean, 0, channel)
            if self.config.probabilistic:
                scale = np.moveaxis(scale, 0, channel)
        else:
            # remove dummy channel dimension altogether
            if self.config.n_channel_out == 1:
                mean = mean[0]
                if self.config.probabilistic:
                    scale = scale[0]

        if normalizer.do_after:
            self.config.n_channel_in == self.config.n_channel_out or _raise(ValueError())
            mean, scale = normalizer.after(mean, scale)

        return mean, scale



class IsotropicCARE(CARE):
    """CARE network for isotropic image reconstruction.

    Extends :class:`csbdeep.models.CARE` by replacing :func:`predict` to do isotropic reconstruction.
    """

    def predict(self, img, factor, normalizer, resizer=PadAndCropResizer(), z=0, channel=None, batch_size=8):
        """Apply neural network to raw image to restore isotropic resolution.

        Parameters
        ----------
        img : :class:`numpy.ndarray`
            Raw input image, with image dimensions expected in the same order as in data for training.
            If applicable, only the z and channel dimensions can be anywhere.
        factor : int
            Upsampling factor for z dimension. It is important that this is chosen in correspondence
            to the subsampling factor used during training data generation.
        normalizer : :class:`csbdeep.predict.Normalizer`
            Normalization of input image before prediction and (potentially) transformation back after prediction.
        resizer : :class:`csbdeep.predict.Resizer`
            If necessary, input image is resized to enable neural network prediction and result is (possibly)
            resized to yield original image size.
        z : int
            Index of z dimension of raw input image.
        channel : int or None
            Index of channel dimension of raw input image. Defaults to ``None``, assuming
            a single-channel input image where without an explicit channel dimension.
        batch_size : int
            Number of image slices that are processed together by the neural network.
            Reduce this value if out of memory errors occur.

        Returns
        -------
        :class:`numpy.ndarray`
            Returns the restored image. If the model is probabilistic, this denotes the `mean` parameter of
            the predicted per-pixel Laplace distributions (i.e., the expected restored image).

        Todo
        ----
        - :func:`scipy.ndimage.interpolation.zoom` differs from :func:`gputools.scale`. Important?

        """
        return self._predict_mean_and_scale(img, factor, normalizer, resizer, z, channel, batch_size)[0]


    def _predict_mean_and_scale(self, img, factor, normalizer, resizer, z=0, channel=None, batch_size=8):
        """Apply neural network to raw image to restore isotropic resolution.

        Parameters
        ----------
        img : :class:`numpy.ndarray`
            Raw input image, with image dimensions expected in the same order as in data for training.
            If applicable, only the z and channel dimensions can be anywhere.
        factor : int
            Upsampling factor for z dimension. It is important that this is chosen in correspondence
            to the subsampling factor used during training data generation.
        normalizer : :class:`csbdeep.predict.Normalizer`
            Normalization of input image before prediction and (potentially) transformation back after prediction.
        resizer : :class:`csbdeep.predict.Resizer`
            If necessary, input image is resized to enable neural network prediction and result is (possibly)
            resized to yield original image size.
        z : int
            Index of z dimension of raw input image.
        channel : int or None
            Index of channel dimension of raw input image. Defaults to ``None``, assuming
            a single-channel input image where without an explicit channel dimension.
        batch_size : int
            Number of image slices that are processed together by the neural network.
            Reduce this value if out of memory errors occur.

        Returns
        -------
        tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray` or None)
            If model is probabilistic, returns a tuple `(mean, scale)` that defines the parameters
            of per-pixel Laplace distributions. Otherwise, returns the restored image via a tuple `(restored,None)`

        Todo
        ----
        - :func:`scipy.ndimage.interpolation.zoom` differs from :func:`gputools.scale`. Important?

        """
        if channel is None:
            self.config.n_channel_in == 1 or _raise(ValueError())
        else:
            -img.ndim <= channel < img.ndim or _raise(ValueError())
            if channel < 0:
                channel %= img.ndim
            self.config.n_channel_in == img.shape[channel] or _raise(ValueError())

        np.isscalar(factor) and factor > 0 or _raise(ValueError())
        channel != z or _raise(ValueError())

        n_channel_predicted = self.config.n_channel_out * (2 if self.config.probabilistic else 1)

        isinstance(resizer,Resizer) or _raise(ValueError())
        isinstance(normalizer,Normalizer) or _raise(ValueError())

        # try gputools for fast scaling function, fallback to scipy
        # problem with gputools: GPU memory can be fully used by tensorflow
        try:
            # raise ImportError
            from gputools import scale as _scale
            def scale_z(arr,factor):
                # gputools.scale can only do 3D arrays
                return np.stack([_scale(_arr,(factor,1,1),interpolation='linear') for _arr in arr])
        except ImportError:
            from scipy.ndimage.interpolation import zoom
            def scale_z(arr,factor):
                return zoom(arr,(1,factor,1,1),order=1,prefilter=False)

        # normalize
        x = normalizer.before(img,channel)

        # move channel and z to front of image
        if channel is None:
            x = np.expand_dims(x,-1)
            _channel = -1
        else:
            _channel = channel
        x = np.moveaxis(x,[_channel,z],[0,1])

        # scale z dimension up (second dimension)
        x_scaled = scale_z(x,factor)

        # resize: make (x,y,z) image dimensions divisible by power of 2 to allow downsampling steps in unet
        div_n = 2 ** self.config.unet_n_depth
        x_scaled = resizer.before(x_scaled,div_n,exclude=0)

        # move channel to the end
        x_scaled = np.moveaxis(x_scaled,0,-1)

        # u1: first rotation and prediction
        x_rot1   = rotate(x_scaled, axis=1, copy=False)
        u_rot1   = predict_direct(self.keras_model, x_rot1, channel_in=-1, channel_out=-1, single_sample=False,
                                  batch_size=batch_size, verbose=0)
        u1       = rotate(u_rot1, -1, axis=1, copy=False)

        # u2: second rotation and prediction
        x_rot2   = rotate(rotate(x_scaled, axis=2, copy=False), axis=0, copy=False)
        u_rot2   = predict_direct(self.keras_model, x_rot2, channel_in=-1, channel_out=-1, single_sample=False,
                                  batch_size=batch_size, verbose=0)
        u2       = rotate(rotate(u_rot2, -1, axis=0, copy=False), -1, axis=2, copy=False)

        u_rot1.shape[-1] == n_channel_predicted or _raise(ValueError())
        u_rot2.shape[-1] == n_channel_predicted or _raise(ValueError())

        # move channel to the front and resize after prediction
        u1 = np.moveaxis(u1, -1, 0)
        u2 = np.moveaxis(u2, -1, 0)
        u1 = resizer.after(u1,exclude=0)
        u2 = resizer.after(u2,exclude=0)

        # combine u1 & u2 and separate mean and scale
        avg = lambda u1,u2: np.sqrt(np.maximum(u1,0)*np.maximum(u2,0)) # geometric mean
        # avg = lambda u1,u2: (u1+u2)/2 # arithmetic mean
        if self.config.probabilistic:
            _n    = self.config.n_channel_out
            mean  =        avg(u1[:_n],u1[:_n])
            scale = np.maximum(u1[_n:],u2[_n:])
        else:
            mean, scale = avg(u1,u2), None

        if channel is None:
            # remove dummy channel dimension altogether
            if self.config.n_channel_out == 1:
                mean = mean[0]
                if self.config.probabilistic:
                    scale = scale[0]

        # move z (and channel) to same dimension as in input image
        _from = 0 if channel is None else [0,      1]
        _to   = z if channel is None else [channel,z]
        mean = np.moveaxis(mean, _from, _to)
        if self.config.probabilistic:
            scale = np.moveaxis(scale, _from, _to)

        if normalizer.do_after:
            self.config.n_channel_in == self.config.n_channel_out or _raise(ValueError())
            mean, scale = normalizer.after(mean, scale)

        return mean, scale

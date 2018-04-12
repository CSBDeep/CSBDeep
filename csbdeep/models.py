from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

import argparse
import datetime

from .utils import _raise, consume, Path
import warnings
import numpy as np
from collections import namedtuple

from . import nets, train


class Config(argparse.Namespace):
    """TODO."""

    def __init__(self, n_dim, n_channel_in=1, n_channel_out=1, probabilistic=False, **kwargs):
        """TODO."""
        n_dim in (2,3) or _raise(ValueError())

        self.n_dim                 = n_dim
        self.n_channel_in          = n_channel_in
        self.n_channel_out         = n_channel_out
        self.probabilistic         = probabilistic
        #
        self.unet_residual         = self.n_channel_in == self.n_channel_out
        self.unet_n_depth          = 2
        self.unet_kern_size        = 5 if self.n_dim==2 else 3
        self.unet_n_first          = 32
        self.unet_last_activation  = 'linear'
        self.unet_input_shape      = self.n_dim*(None,) + (self.n_channel_in,)
        #
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

        # TODO: param checks





class CARE(object):
    """TODO."""

    def __init__(self, config, name=None, outdir='.'):
        """TODO."""
        self.config = config
        self.outdir = Path(outdir)
        self.name = name
        self.keras_model = self._build()
        self._model_prepared = False
        self._set_logdir()


    def _set_logdir(self):
        # if not self.outdir.exists():
        #     self.outdir.mkdir(parents=True, exist_ok=True)
        # self.outdir.is_dir() or _raise(FileNotFoundError())
        if self.name is None:
            self.name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
        self.logdir = self.outdir / self.name
        # self.logdir.mkdir(parents=True, exist_ok=True)
        # print(self.outdir)
        # print(self.logdir)


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
        """TODO."""
        self.keras_model.load_weights(str(self.logdir/name))


    def prepare_for_training(self, optimizer=None, **kwargs):
        """TODO."""
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
        """TODO."""
        if not self._model_prepared:
            self.prepare_for_training()

        if self.logdir.exists():
            warnings.warn('output path for model already exists, files may be overwritten during training: %s' % str(self.logdir.resolve()))

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
        """TODO."""
        from csbdeep.tf import export_SavedModel
        fout = self.logdir / 'TF_SavedModel.zip'
        export_SavedModel(self.keras_model, str(fout))
        print("\nModel exported in TensorFlow's SavedModel format:\n%s" % str(fout.resolve()))


    def predict(self, img, normalization, n_tiles):
        """TODO."""
        raise NotImplementedError()
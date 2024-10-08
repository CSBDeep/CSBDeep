# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, absolute_import, division

import datetime
import warnings
import sys

import numpy as np
from six import string_types, PY2
from functools import wraps

from .config import BaseConfig
from ..utils import _raise, load_json, save_json, axes_check_and_normalize, axes_dict, move_image_axes
from ..utils.six import Path, FileNotFoundError
from ..utils.tf import keras_import, IS_KERAS_3_PLUS
from ..data import Normalizer, NoNormalizer
from ..data import Resizer, NoResizer
from .pretrained import get_model_details, get_model_instance, get_registered_models

from six import add_metaclass
from abc import ABCMeta, abstractmethod, abstractproperty



def suppress_without_basedir(warn):
    def _suppress_without_basedir(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            self = args[0]
            if self.basedir is None:
                warn is False or warnings.warn("Suppressing call of '%s' (due to basedir=None)." % f.__name__)
            else:
                return f(*args, **kwargs)
        return wrapper
    return _suppress_without_basedir



if IS_KERAS_3_PLUS:
    import h5py
    from packaging.version import Version
    from keras.src.legacy.saving import legacy_h5_format
    from keras.src import backend, __version__ as keras_version

    if Version(keras_version) >= Version("3.3.0"):
        save_weights_to_hdf5_group = legacy_h5_format.save_weights_to_hdf5_group
    else:
        def save_weights_to_hdf5_group(f, model):
            legacy_h5_format.save_attributes_to_hdf5_group(f, "layer_names", [layer.name.encode("utf8") for layer in model.layers])
            f.attrs["backend"] = backend.backend().encode("utf8")
            f.attrs["keras_version"] = str(keras_version).encode("utf8")
            for layer in sorted(model.layers, key=lambda x: x.name):
                g = f.create_group(layer.name)
                weights = legacy_h5_format._legacy_weights(layer)
                save_subset_weights_to_hdf5_group(g, weights)
            g = f.create_group("top_level_model_weights")
            weights = [v for v in model._trainable_variables + model._non_trainable_variables if v in model.weights]
            save_subset_weights_to_hdf5_group(g, weights)

        def save_subset_weights_to_hdf5_group(f, weights):
            # FIX: use w.path instead of w.name to avoid name collisions (for "functional" layers)
            # -> has been fixed since keras 3.3.0: https://github.com/keras-team/keras/blob/v3.3.0/keras/src/legacy/saving/legacy_h5_format.py#L234
            weight_names = [w.path.encode("utf8") for w in weights]
            weight_values = [backend.convert_to_numpy(w) for w in weights]
            legacy_h5_format.save_attributes_to_hdf5_group(f, "weight_names", weight_names)
            for name, val in zip(weight_names, weight_values):
                param_dset = f.create_dataset(name, val.shape, dtype=val.dtype)
                param_dset[() if not val.shape else slice(None)] = val

    def _keras3_monkey_patch_legacy_weights(model):
        ref_save_weights = model.save_weights

        def save_weights(self, filepath, overwrite=True):
            p = Path(filepath)
            if not overwrite and p.exists():
                raise FileExistsError(f"Weights file already exists: {str(p.resolve())}")
            if p.name.endswith(".weights.h5"):
                warnings.warn("Detected filename suffix '.weights.h5', thus saving in newer Keras 3.x file format (cannot be loaded with Keras 2.x)")
            if not p.name.endswith(".weights.h5"):
                with h5py.File(str(p), "w") as f:
                    save_weights_to_hdf5_group(f, self)
            else:
                return ref_save_weights(filepath, overwrite=overwrite)

        model.save_weights = save_weights.__get__(model)



@add_metaclass(ABCMeta)
class BaseModel(object):
    """Base model.

    Subclasses must implement :func:`_build` and :func:`_config_class`.

    Parameters
    ----------
    config : Subclass of :class:`csbdeep.models.BaseConfig` or None
        Valid configuration of a model (see :func:`BaseConfig.is_valid`).
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

    Attributes
    ----------
    config : :class:`csbdeep.models.BaseConfig`
        Configuration of the model, as provided during instantiation.
    keras_model : `Keras model <https://keras.io/getting-started/functional-api-guide/>`_
        Keras neural network model.
    name : str
        Model name.
    logdir : :class:`pathlib.Path`
        Path to model folder (which stores configuration, weights, etc.)
    """

    @classmethod
    def from_pretrained(cls, name_or_alias=None):
        try:
            get_model_details(cls, name_or_alias, verbose=True)
            return get_model_instance(cls, name_or_alias)
        except ValueError as e:
            if name_or_alias is not None:
                print("Could not find model with name or alias '%s'" % (name_or_alias), file=sys.stderr)
                sys.stderr.flush()
            get_registered_models(cls, verbose=True)


    def __init__(self, config, name=None, basedir='.'):
        """See class docstring."""

        config is None or isinstance(config,self._config_class) or _raise (
            ValueError("Invalid configuration of type '%s', was expecting type '%s'." % (type(config).__name__, self._config_class.__name__))
        )
        if config is not None and not config.is_valid():
            invalid_attr = config.is_valid(True)[1]
            raise ValueError('Invalid configuration attributes: ' + ', '.join(invalid_attr))
        (not (config is None and basedir is None)) or _raise(ValueError("No config provided and cannot be loaded from disk since basedir=None."))

        name is None or (isinstance(name,string_types) and len(name)>0) or _raise(ValueError("No valid name: '%s'" % str(name)))
        basedir is None or isinstance(basedir,(string_types,Path)) or _raise(ValueError("No valid basedir: '%s'" % str(basedir)))
        self.config = config
        self.name = name if name is not None else datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
        self.basedir = Path(basedir) if basedir is not None else None
        if config is not None:
            # config was provided -> update before it is saved to disk
            self._update_and_check_config()
        self._set_logdir()
        if config is None:
            # config was loaded from disk -> update it after loading
            self._update_and_check_config()
        self._model_prepared = False
        self.keras_model = self._build()
        if IS_KERAS_3_PLUS and isinstance(self.keras_model, keras_import('models', 'Model')):
            # monkey-patch keras model to save weights in legacy format if suffix is not '.weights.h5'
            _keras3_monkey_patch_legacy_weights(self.keras_model)
        if config is None:
            self._find_and_load_weights()


    def __repr__(self):
        s = ("{self.__class__.__name__}({self.name}): {self.config.axes} → {self._axes_out}\n".format(self=self) +
             "├─ Directory: {}\n".format(self.logdir.resolve() if self.basedir is not None else None) +
             self._repr_extra() +
             "└─ {self.config}".format(self=self))
        return s.encode('utf-8') if PY2 else s


    def _repr_extra(self):
        return ""


    def _update_and_check_config(self):
        pass


    @suppress_without_basedir(warn=False)
    def _set_logdir(self):
        self.logdir = self.basedir / self.name

        config_file =  self.logdir / 'config.json'
        if self.config is None:
            if config_file.exists():
                config_dict = load_json(str(config_file))
                config_dict = self._config_class.update_loaded_config(config_dict)
                self.config = self._config_class(**config_dict)
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


    @suppress_without_basedir(warn=False)
    def _find_and_load_weights(self,prefer='best'):
        from itertools import chain
        # get all weight files and sort by modification time descending (newest first)
        weights_ext   = ('*.h5','*.hdf5')
        weights_files = chain(*(self.logdir.glob(ext) for ext in weights_ext))
        weights_files = reversed(sorted(weights_files, key=lambda f: f.stat().st_mtime))
        weights_files = list(weights_files)
        if len(weights_files) == 0:
            warnings.warn("Couldn't find any network weights (%s) to load." % ', '.join(weights_ext))
            return
        weights_preferred = list(filter(lambda f: prefer in f.name, weights_files))
        weights_chosen = weights_preferred[0] if len(weights_preferred)>0 else weights_files[0]
        print("Loading network weights from '%s'." % weights_chosen.name)
        self.load_weights(weights_chosen.name)


    @abstractmethod
    def _build(self):
        """ Create and return a Keras model. """


    @suppress_without_basedir(warn=True)
    def load_weights(self, name='weights_best.h5'):
        """Load neural network weights from model folder.

        Parameters
        ----------
        name : str
            Name of HDF5 weight file (as saved during or after training).
        """
        self.keras_model.load_weights(str(self.logdir/name))


    def _checkpoint_callbacks(self):
        callbacks = []
        if self.basedir is not None:
            from ..utils.tf import keras_import
            ModelCheckpoint = keras_import('callbacks', 'ModelCheckpoint')
            # keras 3: need to add suffix to filename because ModelCheckpoint constructor throws error if it's missing
            suffix = ".weights.h5" if IS_KERAS_3_PLUS else ""
            if self.config.train_checkpoint is not None:
                callbacks.append(ModelCheckpoint(str(self.logdir / self.config.train_checkpoint) + suffix,  save_best_only=True,  save_weights_only=True))
                # keras3: remove suffix because patched model.save_weights can save in legacy format
                if IS_KERAS_3_PLUS: callbacks[-1].filepath = callbacks[-1].filepath[:-len(suffix)]
            if self.config.train_checkpoint_epoch is not None:
                callbacks.append(ModelCheckpoint(str(self.logdir / self.config.train_checkpoint_epoch) + suffix, save_best_only=False, save_weights_only=True))
                # keras3: remove suffix because patched model.save_weights can save in legacy format
                if IS_KERAS_3_PLUS: callbacks[-1].filepath = callbacks[-1].filepath[:-len(suffix)]
        return callbacks


    def _training_finished(self):
        if self.basedir is not None:
            if self.config.train_checkpoint_last is not None:
                self.keras_model.save_weights(str(self.logdir / self.config.train_checkpoint_last))
            if self.config.train_checkpoint is not None:
                print()
                self._find_and_load_weights(self.config.train_checkpoint)
            if self.config.train_checkpoint_epoch is not None:
                try:
                    # remove temporary weights
                    (self.logdir / self.config.train_checkpoint_epoch).unlink()
                except FileNotFoundError:
                    pass


    @suppress_without_basedir(warn=True)
    def export_TF(self, fname=None):
        raise NotImplementedError()


    def _make_permute_axes(self, img_axes_in, net_axes_in, net_axes_out=None, img_axes_out=None):
        # img_axes_in -> net_axes_in ---NN--> net_axes_out -> img_axes_out
        if net_axes_out is None:
            net_axes_out = net_axes_in
        if img_axes_out is None:
            img_axes_out = img_axes_in
        assert 'C' in net_axes_in and 'C' in net_axes_out
        assert not 'C' in img_axes_in or 'C' in img_axes_out

        def _permute_axes(data,undo=False):
            if data is None:
                return None
            if undo:
                if 'C' in img_axes_in:
                    return move_image_axes(data, net_axes_out, img_axes_out, True)
                else:
                    # input is single-channel and has no channel axis
                    data = move_image_axes(data, net_axes_out, img_axes_out+'C', True)
                    if data.shape[-1] == 1:
                        # output is single-channel -> remove channel axis
                        data = data[...,0]
                    return data
            else:
                return move_image_axes(data, img_axes_in, net_axes_in, True)
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


    @property
    def _axes_out(self):
        return self.config.axes


    @abstractproperty
    def _config_class(self):
        """ Class of config to be used for this model. """

from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

import numpy as np
import os
import warnings
import shutil
import datetime

import tensorflow as tf
# from tensorflow.python.framework import graph_util
# from tensorflow.python.framework import graph_io

import keras
from keras import backend as K
from keras.callbacks import Callback
from keras.layers import Lambda

from .utils import _raise, is_tf_dim, is_tf_back, save_json, tempfile



def limit_gpu_memory(fraction, allow_growth=False):
    """Limit GPU memory allocation for TensorFlow (TF) backend.

    Parameters
    ----------
    fraction : float
        Limit TF to use only a fraction (value between 0 and 1) of the available GPU memory.
        Reduced memory allocation can be disabled if fraction is set to ``None``.
    allow_growth : bool, optional
        If ``False`` (default), TF will allocate all designated (see `fraction`) memory all at once.
        If ``True``, TF will allocate memory as needed up to the limit imposed by `fraction`; this may
        incur a performance penalty due to memory fragmentation.

    Raises
    ------
    ValueError
        If `fraction` is not ``None`` or a float value between 0 and 1.
    NotImplementedError
        If TensorFlow is not used as the backend.
    """

    fraction is None or (np.isscalar(fraction) and 0<=fraction<=1) or _raise(ValueError('fraction must be between 0 and 1.'))

    if is_tf_back():
        if K.tensorflow_backend._SESSION is None:
            config = tf.ConfigProto()
            if fraction is not None:
                config.gpu_options.per_process_gpu_memory_fraction = fraction
            config.gpu_options.allow_growth = bool(allow_growth)
            session = tf.Session(config=config)
            K.tensorflow_backend.set_session(session)
            # print("[tf_limit]\t setting config.gpu_options.per_process_gpu_memory_fraction to ",config.gpu_options.per_process_gpu_memory_fraction)
        else:
            warnings.warn('Too late too limit GPU memory, must be done before any computation.')
    else:
        raise NotImplementedError('Not using tensorflow backend.')



def export_SavedModel(model, outpath, meta={}, format='zip'):
    """Export Keras model in TensorFlow's SavedModel_ format.

    See `Your Model in Fiji`_ to learn how to use the exported model with our CSBDeep Fiji plugins.

    .. _SavedModel: https://www.tensorflow.org/programmers_guide/saved_model#structure_of_a_savedmodel_directory
    .. _`Your Model in Fiji`: https://github.com/CSBDeep/CSBDeep/wiki/Your-Model-in-Fiji

    Parameters
    ----------
    model : :class:`keras.Model`
        Keras model to be exported.
    outpath : str
        Path of the file/folder that the model will exported to.
    meta : dict, optional
        Metadata to be saved in an additional ``meta.json`` file.
    format : str, optional
        Can be 'dir' to export as a directory or 'zip' (default) to export as a ZIP file.

    Raises
    ------
    ValueError
        Illegal arguments.

    """

    def export_to_dir(dirname):
        if len(model.input_layers) > 1 or len(model.output_layers) > 1:
            warnings.warn('Not tested with multiple input or output layers.')
        builder = tf.saved_model.builder.SavedModelBuilder(dirname)
        signature = tf.saved_model.signature_def_utils.predict_signature_def(
            inputs  = {'input':  model.input},
            outputs = {'output': model.output}
        )
        signature_def_map = { tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature }
        builder.add_meta_graph_and_variables(K.get_session(),
                                             [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map=signature_def_map)
        builder.save()
        if meta is not None and len(meta) > 0:
            save_json(meta, os.path.join(dirname,'meta.json'))



    ## checks
    isinstance(model,keras.Model) or _raise(ValueError("'model' must be a Keras model."))
    # supported_formats = tuple(['dir']+[name for name,description in shutil.get_archive_formats()])
    supported_formats = 'dir','zip'
    format in supported_formats or _raise(ValueError("Unsupported format '%s', must be one of %s." % (format,str(supported_formats))))

    # remove '.zip' file name extension if necessary
    if format == 'zip' and outpath.endswith('.zip'):
        outpath = os.path.splitext(outpath)[0]

    if format == 'dir':
        export_to_dir(outpath)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpsubdir = os.path.join(tmpdir,'model')
            export_to_dir(tmpsubdir)
            shutil.make_archive(outpath, format, tmpsubdir)




class CARETensorBoard(Callback):
    """ TODO """
    def __init__(self, log_dir='./logs',
                 freq=1,
                 compute_histograms=False,
                 n_images=3,
                 prob_out=False,
                 write_graph=False,
                 prefix_with_timestamp=True,
                 write_images=False):
        super(CARETensorBoard, self).__init__()
        if K.backend() != 'tensorflow':
            raise RuntimeError('TensorBoard callback only works '
                               'with the TensorFlow backend.')

        self.freq = freq
        self.image_freq = freq
        self.prob_out = prob_out
        self.merged = None
        self.write_graph = write_graph
        self.write_images = write_images
        self.n_images = n_images
        self.compute_histograms = compute_histograms

        if prefix_with_timestamp:
            log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f"))

        self.log_dir = log_dir

    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()
        tf_sums = []

        if self.compute_histograms and self.freq and self.merged is None:
            for layer in self.model.layers:
                for weight in layer.weights:
                    tf_sums.append(tf.summary.histogram(weight.name, weight))

                if hasattr(layer, 'output'):
                    tf_sums.append(tf.summary.histogram('{}_out'.format(layer.name),
                                                        layer.output))

        # outputs
        if K.image_dim_ordering() == "tf":
            n_channels_in = self.model.input_shape[-1]
            n_dim_in = len(self.model.input_shape)

            # FIXME: not fully baked, eg. n_dim==5 multichannel doesnt work

            if n_dim_in > 4:
                # print("tensorboard shape: %s"%str(self.model.input_shape))
                input_layer = Lambda(lambda x: K.max(K.max(x, axis=1), axis=-1, keepdims=True))(self.model.input)
            else:
                if n_channels_in > 3:
                    input_layer = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(self.model.input)
                elif n_channels_in == 2:
                    input_layer = Lambda(lambda x: K.concatenate([x,x[...,:1]], axis=-1))(self.model.input)
                else:
                    input_layer = self.model.input

            n_channels_out = self.model.output_shape[-1]
            n_dim_out = len(self.model.output_shape)

            if self.prob_out:
                # first half of output channels is mean, second half scale
                # assert n_channels_in*2 == n_channels_out
                # if n_channels_in*2 != n_channels_out:
                #     raise ValueError('prob_out: must be two output channels for every input channel')
                if n_channels_out % 2 != 0:
                    raise ValueError()

                sep = n_channels_out // 2
            else:
                sep = n_channels_out

            if n_dim_out > 4:
                output_layer = Lambda(lambda x: K.max(K.max(x[...,:sep], axis=1), axis=-1, keepdims=True))(self.model.output)
            else:
                if sep > 3:
                    output_layer = Lambda(lambda x: K.max(x[...,:sep], axis=-1, keepdims=True))(self.model.output)
                elif sep == 2:
                    output_layer = Lambda(lambda x: K.concatenate([x[...,:sep],x[...,:1]], axis=-1))(self.model.output)
                else:
                    output_layer = Lambda(lambda x: x[...,:sep])(self.model.output)

            if self.prob_out:
                # scale images
                if n_dim_out > 4:
                    scale_layer = Lambda(lambda x: K.max(K.max(x[...,sep:], axis=1), axis=-1, keepdims=True))(self.model.output)
                else:
                    if sep > 3:
                        scale_layer = Lambda(lambda x: K.max(x[...,sep:], axis=-1, keepdims=True))(self.model.output)
                    elif sep == 2:
                        scale_layer = Lambda(lambda x: K.concatenate([x[...,sep:],x[...,-1:]], axis=-1))(self.model.output)
                    else:
                        scale_layer = Lambda(lambda x: x[...,sep:])(self.model.output)




        else:
            raise NotImplementedError()
        #
        #     n_channels = self.model.input_shape[0]
        #     if n_channels > 1:
        #         #input_layer  = Lambda(lambda x: x[n_channels // 2:n_channels // 2 + 1, :, :, :])(self.model.input)
        #         input_layer = Lambda(lambda x: K.max(x, axis = 0))(self.model.input)
        #     else:
        #         input_layer = self.model.input
        # #
        # if K.image_dim_ordering() == "tf":
        #     n_channels = self.model.input_shape[-1]
        #     if n_channels>1:
        #         input_layer = Lambda(lambda x: x[:, :, :, n_channels // 2:n_channels // 2 + 1])(self.model.input)
        #     else:
        #         input_layer = self.model.input
        # else:
        #     n_channels = self.model.input_shape[0]
        #     if n_channels > 1:
        #         #input_layer  = Lambda(lambda x: x[n_channels // 2:n_channels // 2 + 1, :, :, :])(self.model.input)
        #         input_layer = Lambda(lambda x: K.max(x, axis = 0))(self.model.input)
        #     else:
        #         input_layer = self.model.input

        tf_sums.append(tf.summary.image('input', input_layer, max_outputs=self.n_images))
        if self.prob_out:
            tf_sums.append(tf.summary.image('mean', output_layer, max_outputs=self.n_images))
            tf_sums.append(tf.summary.image('scale', scale_layer, max_outputs=self.n_images))
        else:
            tf_sums.append(tf.summary.image('output', output_layer, max_outputs=self.n_images))

        with tf.name_scope('merged'):
            self.merged = tf.summary.merge(tf_sums)
            # self.merged = tf.summary.merge([foo])

        with tf.name_scope('summary_writer'):
            if self.write_graph:
                self.writer = tf.summary.FileWriter(self.log_dir,
                                                    self.sess.graph)
            else:
                self.writer = tf.summary.FileWriter(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if self.validation_data and self.freq:
            if epoch % self.freq == 0:
                # TODO: implement batched calls to sess.run
                # (current call will likely go OOM on GPU)
                if self.model.uses_learning_phase:
                    cut_v_data = len(self.model.inputs)
                    val_data = self.validation_data + [0]
                    tensors = self.model.inputs + [K.learning_phase()]
                else:
                    val_data = list(v[:self.n_images] for v in self.validation_data)
                    tensors = self.model.inputs
                feed_dict = dict(zip(tensors, val_data))
                result = self.sess.run([self.merged], feed_dict=feed_dict)
                summary_str = result[0]

                self.writer.add_summary(summary_str, epoch)

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def on_train_end(self, _):
        self.writer.close()

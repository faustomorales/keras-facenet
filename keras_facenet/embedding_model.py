
import logging
import zipfile
import shutil
import os
import re

import tensorflow as tf
import numpy as np

from .inception_resnet_v1 import InceptionResNetV1
from .utils import sha256sum, download_and_verify

log = logging.getLogger(__name__)

# regex for renaming the tensors to their corresponding Keras counterpart
RE_REPEAT = re.compile(r'Repeat_[0-9_]*b')
RE_BLOCK8 = re.compile(r'Block8_[A-Za-z]')


def get_filename(key):
    filename = str(key)
    filename = filename.replace('/', '_')
    filename = filename.replace('InceptionResnetV1_', '')

    # remove "Repeat" scope from filename
    filename = RE_REPEAT.sub('B', filename)

    if RE_BLOCK8.match(filename):
        # the last block8 has different name with the previous 5 occurrences
        filename = filename.replace('Block8', 'Block8_6')

    # from TF to Keras naming
    filename = filename.replace('_weights', '_kernel')
    filename = filename.replace('_biases', '_bias')

    return filename + '.npy'


def verify_files(metadata, cache_folder):
    for k, v in metadata['files'].items():
        filepath = os.path.join(cache_folder, metadata['dir_name'], v['name'])
        if not os.path.isfile(filepath) or sha256sum(filepath) != v['sha256']:
            return False
    return True


def download_tf_model(metadata, cache_folder):
    """Get TensorFlow model files.
    """
    os.makedirs(cache_folder, exist_ok=True)
    zip_url = metadata['zip_url']
    zip_path = os.path.join(cache_folder, metadata['zip_local_name'])
    dir_path = os.path.join(cache_folder, metadata['dir_name'])
    download_and_verify(
        url=zip_url, filepath=zip_path, sha256=metadata.get('zip_sha256')
    )
    if not os.path.isdir(dir_path) or not verify_files(metadata, cache_folder):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            log.info('Extracting files.')
            zip_ref.extractall(cache_folder)
    assert verify_files(metadata, cache_folder), 'Error occurred verifying file hashes.'  # noqa: E501


def get_keras_model_from_tensorflow(metadata, cache_folder):
    if not verify_files(metadata, cache_folder):
        download_tf_model(metadata, cache_folder)
    model_dir = os.path.join(cache_folder, metadata['dir_name'])
    npy_weights_dir = os.path.join(model_dir, 'npy')
    model_dir = model_dir

    os.makedirs(npy_weights_dir, exist_ok=True)

    weights_filename = metadata['keras_weights_filename']
    model_filename = metadata['keras_model_filename']

    log.info('Loading TensorFlow weights.')
    reader = tf.train.NewCheckpointReader(
        os.path.join(model_dir, metadata['reader_prefix'])
    )
    for key in reader.get_variable_to_shape_map():
        # not saving the following tensors
        if key == 'global_step':
            continue
        if 'AuxLogit' in key:
            continue

        # convert tensor name into the corresponding Keras layer weight name
        # and save
        path = os.path.join(npy_weights_dir, get_filename(key))
        arr = reader.get_tensor(key)
        np.save(path, arr)

    log.info('Building Inception model.')
    model = InceptionResNetV1(
        input_shape=(None, None, 3),
        classes=metadata['dimensions']
    )

    log.info('Loading numpy weights from ' + npy_weights_dir)
    for layer in model.layers:
        if layer.weights:
            weights = []
            for w in layer.weights:
                log.info('Loading weights for ' + layer.name)
                weight_name = os.path.basename(w.name).replace(':0', '')
                weight_file = layer.name + '_' + weight_name + '.npy'
                weight_arr = np.load(os.path.join(npy_weights_dir, weight_file))  # noqa: E501
                weights.append(weight_arr)
            layer.set_weights(weights)

    log.info('Saving weights...')
    model.save_weights(os.path.join(model_dir, weights_filename))
    log.info('Saving model...')
    model.save(os.path.join(model_dir, model_filename))
    log.info('Cleaning up numpy weights...')
    shutil.rmtree(npy_weights_dir)
    return model


def get_keras_model_from_prebuilt(metadata, cache_folder):
    log.info('Loading weights.')
    weights_filepath = os.path.join(
        cache_folder,
        metadata['dir_name'],
        metadata['keras_weights_filename']
    )
    download_and_verify(
        url=metadata['keras_weights_url'],
        filepath=weights_filepath,
        sha256=metadata['keras_weights_sha256']
    )
    model = InceptionResNetV1(
        input_shape=(None, None, 3),
        classes=metadata['dimensions']
    )
    model.load_weights(weights_filepath)
    return model

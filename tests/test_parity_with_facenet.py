import os
import logging
import tempfile
import subprocess

import tensorflow as tf
import numpy as np

import keras_facenet

logging.basicConfig()
log = logging.getLogger()
log.setLevel('INFO')

facenet_data_dir = 'tests/facenet_data_dir'
keras_facenet_data_dir = 'tests/keras_facenet_data_dir'
os.makedirs(facenet_data_dir, exist_ok=True)
os.makedirs(keras_facenet_data_dir, exist_ok=True)
images = np.random.randn(3, 160, 160, 3)

subprocess.call(['git', 'submodule', 'init'])
subprocess.call(['git', 'submodule', 'update'])

from .facenet.src import download_and_extract, facenet  # noqa: E402

def load_model(model):
    # This is a modified version of facenet.load_model to support TensorFlow 2.x.
    model_exp = os.path.expanduser(model)
    meta_file, ckpt_file = facenet.get_model_filenames(model_exp)    
    saver = tf.compat.v1.train.import_meta_graph(os.path.join(model_exp, meta_file))
    saver.restore(tf.compat.v1.get_default_session(), os.path.join(model_exp, ckpt_file))


def run_test(model_name):
    download_and_extract.download_and_extract_file(
        model_name=model_name,
        data_dir=facenet_data_dir
    )
    kfn = keras_facenet.FaceNet(
        key=model_name,
        cache_folder=keras_facenet_data_dir
    )
    emb_kfn = kfn.embeddings(images)
    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            load_model(os.path.join(facenet_data_dir, model_name))
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")  # noqa: E501
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")  # noqa: E501
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")  # noqa: E501
            if kfn.metadata['fixed_image_standardization']:
                X = (images - 127.5) / 127.5
            else:
                X = np.array([facenet.prewhiten(image) for image in images])
            feed_dict = {images_placeholder: X, phase_train_placeholder: False}  # noqa: E501
            emb = sess.run(embeddings, feed_dict=feed_dict)
    print('Comparing results for', model_name)
    np.testing.assert_almost_equal(emb, emb_kfn)


def test_2017_weights():
    run_test('20170512-110547')


def test_2018_weights():
    run_test('20180402-114759')
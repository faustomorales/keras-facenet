import logging
import os

from scipy import spatial
import tensorflow as tf
import cv2

from . import embedding_model, metadata
import numpy as np

log = logging.getLogger(__name__)


class FaceNet:
    """An object wrapping the FaceNet embedding model.

    Args:
        key: Which version of the model to use. Options
            are: 20180402-114759, 20180408-102900, 20170511-185253,
            and 20170512-110547
        use_prebuilt: Whether to use a prebuilt Keras model. If False,
            a Keras model is build from the TensorFlow protobuf files
            provided by David Sandberg (see
            https://github.com/davidsandberg/facenet for details)
        cache_folder: Where to save and look for model weights (defaults
            to ~/.keras-facenet)

    Attributes:
        metadata: The metadata for the selected model
        cache_folder: The cache folder for the wrapper
        emb
    """
    def __init__(
        self,
        key='20180402-114759',
        use_prebuilt=True,
        cache_folder='~/.keras-facenet'
    ):
        if key not in metadata.MODEL_METADATA:
            raise NotImplementedError('Did not recognize key: ' + key)
        self.metadata = metadata.MODEL_METADATA[key]
        self.cache_folder = os.path.expanduser(cache_folder)
        if use_prebuilt:
            builder = embedding_model.get_keras_model_from_prebuilt
        else:
            builder = embedding_model.get_keras_model_from_tensorflow
        self.model = builder(self.metadata, self.cache_folder)

    def _normalize(self, image):
        if self.metadata['fixed_image_standardization']:
            return (np.float32(image) - 127.5) / 127.5
        else:
            return tf.image.per_image_standardization(image)

    def embeddings(self, images):
        """Compute embeddings for a set of images.

        Args:
            images: A list of images (cropped faces)

        Returns:
            Embeddings of shape (N, K) where N is the
            number of cropepd faces and K is the dimensionality
            of the selected model.
        """
        s = self.metadata['image_size']
        images = [cv2.resize(image, (s, s)) for image in images]
        X = np.float32([self._normalize(image) for image in images])
        embeddings = self.model.predict(X)
        return embeddings

    def compute_distance(self, embedding1, embedding2):
        """Compute the distance between two embeddings.

        Args:
            embedding1: The first embedding
            embedding2: The second embedding

        Returns:
            The distance between the two embeddings.
        """
        if self.metadata['distance_metric'] == 'cosine':
            return spatial.distance.cosine(embedding1, embedding2)
        elif self.metadata['distance_metric'] == 'euclidean':
            return spatial.distance.euclidean(embedding1, embedding2)
        else:
            raise NotImplementedError

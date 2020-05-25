import logging
import os

from scipy import spatial
import cv2

from . import embedding_model, metadata, utils
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
            mean = np.mean(image)
            std = np.std(image)
            std_adj = np.maximum(std, 1.0/np.sqrt(image.size))
            y = np.multiply(np.subtract(image, mean), 1/std_adj)
            return y

    @classmethod
    def mtcnn(cls):
        if not hasattr(cls, '_mtcnn'):
            from mtcnn.mtcnn import MTCNN
            cls._mtcnn = MTCNN()
        return cls._mtcnn
    
    def crop(self, filepath_or_image, threshold=0.95):
        """Get face crops from images.

        Args:
            filepath_or_image: The input image (see extract)
            threshold: The threshold to use for face detection
        """
        if isinstance(filepath_or_image, str):
            image = cv2.imread(filepath_or_image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = filepath_or_image
        detections = [detection for detection in self.mtcnn().detect_faces(image) if detection['confidence'] > threshold]
        if not detections:
            return [], []
        margin = int(0.1*self.metadata['image_size'])
        crops = [utils.cropBox(image, detection=d, margin=margin) for d in detections]
        return detections, crops

    def extract(self, filepath_or_image, threshold=0.95):
        """Extract faces and compute embeddings in one go. Requires
        mtcnn to be installed.

        Args:
            filepath_or_image: Path to image (or an image as RGB array)
            threshold: The threshold for a face to be considered
        Returns:
            Same output as `mtcnn.MTCNN.detect_faces()` but enriched
            with an "embedding" vector.
        """
        detections, crops = self.crop(filepath_or_image, threshold=threshold)
        if not detections:
            return []
        return [{**d, 'embedding': e} for d, e in zip(detections, self.embeddings(images=crops))]

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

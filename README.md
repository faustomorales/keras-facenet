# keras-facenet

This is a simple wrapper around [this wonderful implementation of FaceNet](https://github.com/davidsandberg/facenet). I wanted something that could be used in other applications, that could use any of the four trained models provided in the linked repository, and that took care of all the setup required to get weights and load them. I prefer using Keras wherever possible because of its API, so I used the [example provided here](https://github.com/nyoki-mtl/keras-facenet) and implemented it as part of the code.

Enough background -- so how are you supposed to use this?

## Installing
```
pip install keras-facenet
```

## Usage
To get embeddings for the faces in an image, you can do something like the following.

```
from keras_facenet import FaceNet
embedder = FaceNet()

# Gets a detection dict for each face
# in an image. Each one has the bounding box and
# face landmarks (from mtcnn.MTCNN) along with
# the embedding from FaceNet.
detections = embedder.extract(image, threshold=0.95)

# If you have pre-cropped images, you can skip the
# detection step.
embeddings = embedder.embeddings(images)
```


### Logging
To see what's going on under the hood, set logging to view `INFO` logs. If using in a Jupyter notebook, you can use the following.

```
import logging

logging.basicConfig()
log = logging.getLogger()
log.setLevel('INFO')
```
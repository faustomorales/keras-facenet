# keras-facenet

This is a simple wrapper around [this wonderful implementation of FaceNet](https://github.com/davidsandberg/facenet). I wanted something that could be used in other applications, that could use any of the four trained models provided in the linked repository, and that took care of all the setup required to get weights and load them. I prefer using Keras wherever possible because of its API, so I used the [example provided here](https://github.com/nyoki-mtl/keras-facenet) and implemented it as part of the code.

Enough background -- so how are you supposed to use this?

## Installing
```
pip install keras-facenet
```

## Usage
To get the embeddings for cropped images of some faces, you can do something like the following.

```
from keras_facenet import FaceNet
embedder = FaceNet()

# images is a list of images, each as an
# np.ndarray of shape (H, W, 3).
embeddings = embedder.embeddings(images)
```

`keras-facenet` expects you to provide cropped images of faces and does not ship with a face detector. You can use another library of your choice to get those lovely cropped images. I provide two examples below.

### Using `keras-facenet` with `mira`
`mira` is another package I developed to do simple object detection. Install it using `pip install mira`

Then you can use `mira` to extract faces with the built-in [MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf) model.

```
from mira.detectors import MTCNN
from keras_facenet import FaceNet

detector = MTCNN()
embedder = FaceNet()

faces = detector.detect(image)
embeddings = embedder.embeddings([
    face.selection.extract(image) for face in faces
])
```

### Using `keras-facenet` with `face_recognition`
`face_recognition` is a fantastic all-in-one package for face detection and recognition. Anecdotally, I find that its face detection model is not quite as good as MTCNN and that the embeddings are not quite as good as FaceNet. but you can use its detection model with FaceNet as follows.

```
from face_recognition import face_location
from keras_facenet import 

embedder = FaceNet()

faces = face_locations(image)
embeddings = embedder.embeddings([
    image[t:b, l:r] for for t, r, b, l in faces
])
```

### Logging
To see what's going on under the hood, set logging to view `INFO` logs. If using in a Jupyter notebook, you can use the following.

```
import logging

logging.basicConfig()
log = logging.getLogger()
log.setLevel('INFO')
```
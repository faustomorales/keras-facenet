import urllib.request
import hashlib
import logging
import os

import numpy as np
import cv2

log = logging.getLogger(__name__)


def download_and_verify(url, filepath, sha256=None):
    os.makedirs(os.path.split(filepath)[0], exist_ok=True)
    log.info('Looking for ' + filepath)
    if not os.path.isfile(filepath) or (sha256
                                        and sha256sum(filepath) != sha256):
        log.info('Downloading ' + filepath)
        urllib.request.urlretrieve(url, filepath)
    assert sha256 == sha256sum(filepath), 'Error occurred verifying sha256.'


def sha256sum(filename):
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


def warpBox(image,
            detection,
            target_height,
            target_width,
            marginX=0,
            marginY=0,
            fill=False
           ):
    """Warp a boxed region in an image given by a rotated rectange (in
    clockwise order starting from top left) into a rectangle with a
    specified width and height.
    Args:
        image: The image from which to take the box
        box: A list of four points starting in the top left
            corner and moving clockwise.
        target_height: The height of the output rectangle
        target_width: The width of the output rectangle
        margin: The margin to put around the box
    """
    x1, y1 = detection['keypoints']['left_eye']
    x2, y2 = detection['keypoints']['right_eye']
    w = np.sqrt(np.square(x2 - x1) + np.square(y2 - y1))
    m = (y2 - y1) / (x2 - x1)
    xm = (detection['keypoints']['mouth_right'][0] + detection['keypoints']['mouth_left'][0]) / 2
    ym = (detection['keypoints']['mouth_right'][1] + detection['keypoints']['mouth_left'][1]) / 2
    c = np.sqrt(1 / (1 + np.square(m)))
    x3 = xm + (w/2)*c
    x4 = xm - (w/2)*c
    y3 = ym + m*(w/2)*c
    y4 = ym - m*(w/2)*c
    box = np.array([
        [x1, y1],
        [x2, y2],
        [x3, y3],
        [x4, y4]
    ])
    h = np.sqrt(np.square(x2 - x3) + np.square(y2 - y3))
    cval = (0, 0, 0)
    (x1, y1), (x2, y2), (x3, y3) = box[:3]
    if fill:
        transform_width = target_width
        transform_height = target_height
    else:
        scale = min(target_width / w, target_height / h)
        transform_width = scale * w
        transform_height = scale * h
    M = cv2.getPerspectiveTransform(src=box.astype('float32'),
                                    dst=np.array([[marginX, marginY], [transform_width - marginX, marginY],
                                                  [transform_width - marginX, transform_height - marginY],
                                                  [marginX, transform_height - marginY]]).astype('float32'))
    crop = cv2.warpPerspective(image, M, dsize=(int(transform_width), int(transform_height)))
    target_shape = (target_height, target_width, 3)
    full = (np.zeros(target_shape) + cval).astype('uint8')
    full[:crop.shape[0], :crop.shape[1]] = crop
    return full
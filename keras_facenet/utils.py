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

def cropBox(image, detection, margin):
    x1, y1, w, h = detection['box']
    x1 -= margin
    y1 -= margin
    w += 2*margin
    h += 2*margin
    if x1 < 0:
        w += x1
        x1 = 0
    if y1 < 0:
        h += y1
        y1 = 0
    return image[y1:y1+h, x1:x1+w]
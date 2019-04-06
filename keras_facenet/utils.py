import urllib.request
import hashlib
import logging
import os

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

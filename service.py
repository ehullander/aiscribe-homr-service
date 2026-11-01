"""CPU-only HTTP boundary for the extracted HOMR structural detector.

SPDX-License-Identifier: AGPL-3.0-or-later
"""
from hashlib import sha256
from hmac import compare_digest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from time import perf_counter
from uuid import uuid4

from PIL import Image, UnidentifiedImageError

MAX_BYTES = 32 * 1024 * 1024
MAX_PIXELS = 24_000_000
MODEL_SHA256 = '6ed36640db4ef5d223098b6d5efe4eda97c66b24a2c72faab8a018c749003a8d'


class RequestError(Exception):
    def __init__(self, status, code):
        self.status, self.code = status, code


class StructuralService:
    def __init__(self, token, model, detector=None, source_url=''):
        if not token or len(token) < 32:
            raise ValueError('A private service credential is required')
        self.token = token
        self.model = Path(model)
        self.detector = detector
        self.source_url = source_url
        self.lock = Lock()
        self.worker_id = uuid4().hex[:12]
        self.calls = 0

    def detect(self, authorization, data, expected_hash=None):
        received = perf_counter()
        if not compare_digest(authorization or '', 'Bearer ' + self.token):
            raise RequestError(401, 'unauthorized')
        if not data or len(data) > MAX_BYTES:
            raise RequestError(413, 'invalid_image_size')
        digest = sha256(data).hexdigest()
        if expected_hash and not compare_digest(expected_hash, digest):
            raise RequestError(400, 'image_identity_mismatch')
        try:
            with Image.open(BytesIO(data)) as image:
                if image.format != 'PNG' or image.width * image.height > MAX_PIXELS or max(image.size) > 16000:
                    raise RequestError(400, 'invalid_png')
                image.verify()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise RequestError(400, 'invalid_png') from None
        started = perf_counter()
        # Model sessions and the bounded geometry cache are shared per process.
        # Serialise inference per CPU instance; Azure scales independent instances.
        with self.lock:
            acquired = perf_counter()
            first_call = self.calls == 0
            with TemporaryDirectory(prefix='homr-') as directory:
                if self.detector is None:
                    from structural.detector import detect
                    self.detector = detect
                path = Path(directory) / 'source.png'
                path.write_bytes(data)  # Exact uploaded PNG, never recompressed or resized here.
                detector_started = perf_counter()
                result = self.detector(path, model_path=self.model)
                detector_seconds = perf_counter()-detector_started
            self.calls += 1
        result = dict(result, source='uploaded-page', source_sha256=digest,
                      runtime='azure-cpu', service_seconds=round(perf_counter()-started, 6),
                      source_url=self.source_url, worker_id=self.worker_id,
                      service_timing=dict(validation_seconds=round(started-received,6),
                          lock_wait_seconds=round(acquired-started,6),
                          detector_seconds=round(detector_seconds,6), first_call=first_call))
        return result

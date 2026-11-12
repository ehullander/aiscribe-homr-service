"""HOMR structural service. SPDX-License-Identifier: AGPL-3.0-or-later."""
import json
import gzip
import os
import logging
import hashlib
import socket
import re
from pathlib import Path
from threading import Lock, Thread
from time import perf_counter, time
import azure.functions as func
from service import StructuralService, RequestError, MODEL_SHA256

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
_service = None
_lock = Lock()
_module_loaded_at = time()
_build_info = Path(__file__).with_name('build-info.json')


def source_url():
    try:
        commit = json.loads(_build_info.read_text())['commit']
    except FileNotFoundError:
        return os.environ.get('HOMR_SOURCE_URL', '')
    if not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('Invalid packaged source identity')
    return 'https://github.com/ehullander/aiscribe-homr-service/tree/' + commit


def instance_id():
    # An opaque machine identity, distinct from StructuralService's process ID.
    raw = os.environ.get('WEBSITE_INSTANCE_ID') or os.environ.get('WEBSITE_ROLE_INSTANCE_ID') or socket.gethostname()
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def service():
    global _service
    with _lock:
        if _service is None:
            _service = StructuralService(os.environ.get('HOMR_SERVICE_TOKEN'),
                Path(__file__).parent / 'models' / 'segnet.onnx',
                source_url=source_url())
    return _service


@app.route(route='health', methods=['GET'])
def health(req):
    return func.HttpResponse(json.dumps(dict(service='homr-structural', model_sha256=MODEL_SHA256,
        device='cpu', source_url=source_url(),
        model_ready=bool(_service and _service.model_ready),
        worker_id=_service.worker_id if _service else None,
        instance_id=instance_id(), module_loaded_at=_module_loaded_at)), mimetype='application/json')


@app.route(route='source', methods=['GET'])
def source(req):
    url = source_url()
    return func.HttpResponse(json.dumps(dict(license='AGPL-3.0-or-later', corresponding_source=url)),
                             mimetype='application/json')


@app.route(route='detect', methods=['POST'])
def detect(req):
    received_at = time()
    entered = perf_counter()
    try:
        detector = service()
        factory_finished = perf_counter()
        body = req.get_body()
        body_finished = perf_counter()
        result = detector.detect(req.headers.get('Authorization'), body,
                                 req.headers.get('X-Source-SHA256'))
        result = {**result, 'service_timing': {**result.get('service_timing', {}),
            'http_entry_at': received_at, 'module_loaded_at': _module_loaded_at,
            'instance_id': instance_id(),
            'factory_seconds': round(factory_finished-entered, 6),
            'body_seconds': round(body_finished-factory_finished, 6)}}
        started=perf_counter()
        data=json.dumps(result, allow_nan=False).encode()
        headers={'X-AIScribe-JSON-Bytes':str(len(data))}
        # The new client advertises this explicitly; older clients continue to
        # receive the same uncompressed JSON representation.
        if req.headers.get('Accept-Encoding','').strip().lower()=='gzip' and len(data)>4096:
            data=gzip.compress(data,compresslevel=3)
            headers['Content-Encoding']='gzip'
        headers['X-AIScribe-Wire-Bytes']=str(len(data))
        headers['X-AIScribe-Encode-Ms']=str(round((perf_counter()-started)*1000,3))
        return func.HttpResponse(data, mimetype='application/json',headers=headers)
    except RequestError as error:
        return func.HttpResponse(json.dumps({'error': error.code}), status_code=error.status,
                                 mimetype='application/json')
    except ValueError:
        # A page with no supported staff geometry is a detector limitation, not
        # a transport failure. The caller retains Apple evidence and source.
        return func.HttpResponse('{"error":"no_supported_geometry"}', status_code=422,
                                 mimetype='application/json')


def preload():
    try:service().prepare()
    except Exception:
        # A failed warmup is visible but does not prevent health/status or the
        # normal first-request path from retrying the same pinned model load.
        logging.warning('homr_model_preload_failed')


if os.environ.get('HOMR_PRELOAD_MODEL')=='1':
    Thread(target=preload,name='homr-preload',daemon=True).start()

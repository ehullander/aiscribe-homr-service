"""HOMR structural service. SPDX-License-Identifier: AGPL-3.0-or-later."""
import json
import os
from pathlib import Path
from threading import Lock
import azure.functions as func
from service import StructuralService, RequestError, MODEL_SHA256

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
_service = None
_lock = Lock()


def service():
    global _service
    with _lock:
        if _service is None:
            _service = StructuralService(os.environ.get('HOMR_SERVICE_TOKEN'),
                Path(__file__).parent / 'models' / 'segnet.onnx',
                source_url=os.environ.get('HOMR_SOURCE_URL', ''))
    return _service


@app.route(route='health', methods=['GET'])
def health(req):
    return func.HttpResponse(json.dumps(dict(service='homr-structural', model_sha256=MODEL_SHA256,
        device='cpu', source_url=os.environ.get('HOMR_SOURCE_URL', ''))), mimetype='application/json')


@app.route(route='source', methods=['GET'])
def source(req):
    url = os.environ.get('HOMR_SOURCE_URL', '')
    return func.HttpResponse(json.dumps(dict(license='AGPL-3.0-or-later', corresponding_source=url)),
                             mimetype='application/json')


@app.route(route='detect', methods=['POST'])
def detect(req):
    try:
        result = service().detect(req.headers.get('Authorization'), req.get_body(),
                                  req.headers.get('X-Source-SHA256'))
        return func.HttpResponse(json.dumps(result, allow_nan=False), mimetype='application/json')
    except RequestError as error:
        return func.HttpResponse(json.dumps({'error': error.code}), status_code=error.status,
                                 mimetype='application/json')
    except ValueError:
        # A page with no supported staff geometry is a detector limitation, not
        # a transport failure. The caller retains Apple evidence and source.
        return func.HttpResponse('{"error":"no_supported_geometry"}', status_code=422,
                                 mimetype='application/json')

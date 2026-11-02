"""Private, temporary HOMR runtime grid endpoint. AGPL-3.0-or-later."""
import json
import os
from pathlib import Path
import azure.functions as func
from runtime_benchmark import RuntimeBenchmark
from service import RequestError

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
benchmark = RuntimeBenchmark(os.environ.get('HOMR_BENCHMARK_TOKEN'),
    Path(__file__).parent / 'models' / 'segnet.onnx')


@app.route(route='benchmark', methods=['POST'])
def run(req):
    try:
        result = benchmark.run(req.headers.get('Authorization'), req.get_body(),
            req.headers.get('X-Source-SHA256'),
            int(req.headers.get('X-Benchmark-Threads', '2')),
            int(req.headers.get('X-Benchmark-Batch', '2')))
        return func.HttpResponse(json.dumps(result, allow_nan=False), mimetype='application/json')
    except RequestError as error:
        return func.HttpResponse(json.dumps({'error': error.code}), status_code=error.status,
            mimetype='application/json')
    except ValueError:
        return func.HttpResponse('{"error":"invalid_input_or_unsupported_geometry"}', status_code=422,
            mimetype='application/json')


@app.route(route='source', methods=['GET'])
def source(req):
    return func.HttpResponse(json.dumps(dict(license='AGPL-3.0-or-later',
        corresponding_source=os.environ.get('HOMR_SOURCE_URL', ''))), mimetype='application/json')

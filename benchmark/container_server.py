"""TLS-only adapter for the disposable CPU benchmark. AGPL-3.0-or-later."""
import base64
from hmac import compare_digest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import ssl
from tempfile import TemporaryDirectory

from runtime_benchmark import RuntimeBenchmark
from service import MAX_BYTES, RequestError


def host_details():
    info = {'cpu_count': os.cpu_count(), 'cpu_affinity': sorted(os.sched_getaffinity(0))}
    for line in Path('/proc/cpuinfo').read_text().splitlines():
        if line.startswith('model name'):
            info['model_name'] = line.split(':', 1)[1].strip()
            break
    for name in ('cpu.max', 'memory.max', 'cpu/cpu.cfs_quota_us',
                 'cpu/cpu.cfs_period_us', 'memory/memory.limit_in_bytes'):
        path = Path('/sys/fs/cgroup') / name
        if path.is_file():
            info[name] = path.read_text().strip()
    return info


def handler_for(benchmark, token, source_url, host):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):
            pass  # Never log request headers, credentials or chart data.

        def reply(self, status, body, close=False):
            data = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            if close:
                self.send_header('Connection', 'close')
                self.close_connection = True
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/api/source':
                self.reply(200, {'license': 'AGPL-3.0-or-later', 'corresponding_source': source_url})
            else:
                self.reply(404, {'error': 'not_found'})

        def do_POST(self):
            if self.path != '/api/benchmark':
                self.reply(404, {'error': 'not_found'}, close=True)
                return
            if not compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + token):
                self.reply(401, {'error': 'unauthorized'}, close=True)
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if self.headers.get('Transfer-Encoding') or not 0 < size <= MAX_BYTES:
                    raise RequestError(413, 'invalid_image_size')
                self.connection.settimeout(30)
                body = self.rfile.read(size)
                if len(body) != size:
                    raise RequestError(400, 'incomplete_body')
                result = benchmark.run(self.headers.get('Authorization'), body,
                    self.headers.get('X-Source-SHA256'),
                    int(self.headers.get('X-Benchmark-Threads', '2')),
                    int(self.headers.get('X-Benchmark-Batch', '2')))
                result['host'] = host
                self.reply(200, result)
            except RequestError as error:
                self.reply(error.status, {'error': error.code}, close=True)
            except (ValueError, TimeoutError):
                self.reply(422, {'error': 'invalid_input_or_unsupported_geometry'}, close=True)
            except Exception as error:
                print('Benchmark handler failed:', type(error).__name__, flush=True)
                self.reply(500, {'error': 'benchmark_failed'}, close=True)
    return Handler


def main():
    token = os.environ['HOMR_BENCHMARK_TOKEN']
    benchmark = RuntimeBenchmark(token, Path(__file__).parent / 'models/segnet.onnx')
    host = host_details()
    with TemporaryDirectory(prefix='benchmark-tls-') as folder:
        paths = []
        for name, variable in [('cert.pem', 'HOMR_TLS_CERT'), ('key.pem', 'HOMR_TLS_KEY')]:
            path = Path(folder) / name
            with path.open('xb') as stream:
                path.chmod(0o600)
                stream.write(base64.b64decode(os.environ[variable], validate=True))
            paths.append(path)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(*paths)
        server = ThreadingHTTPServer(('0.0.0.0', 8443), handler_for(
            benchmark, token, os.environ['HOMR_SOURCE_URL'], host))
        server.socket = context.wrap_socket(server.socket, server_side=True)
        print('CPU benchmark ready', json.dumps(host), flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()

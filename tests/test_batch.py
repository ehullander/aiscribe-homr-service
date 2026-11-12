import base64
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import azure.functions as func
from PIL import Image
import function_app as http
from service import StructuralService, RequestError
from structural import batch

TOKEN = 'x' * 40
AUTH = 'Bearer ' + TOKEN


def png(color):
    output = BytesIO()
    Image.new('RGB', (31, 43), color).save(output, format='PNG')
    return output.getvalue()


def page(identity, data):
    return dict(id=identity, png=base64.b64encode(data).decode(),
                sha256=hashlib.sha256(data).hexdigest())


def envelope(*items):
    return json.dumps(dict(version=1, pages=list(items))).encode()


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b = png('black'), png('white')
        self.seen = []
        def detector(path, model_path):
            data = path.read_bytes()
            self.seen.append(data)
            return dict(systems=[dict(top=1.25, bottom=39.75)],
                        notehead_boxes=[dict(cx=17.125, cy=22.625)],
                        source=str(path))
        self.detector = StructuralService(TOKEN, 'unused.onnx', detector=detector)

    def test_exact_original_bytes_per_page_and_geometry_without_offsets(self):
        result = batch.detect(self.detector, AUTH, envelope(page('one', self.a), page('two', self.b)))
        self.assertEqual(self.seen, [self.a, self.b])
        self.assertEqual(result['status'], 'complete')
        self.assertEqual([x['id'] for x in result['pages']], ['one', 'two'])
        for index, item in enumerate(result['pages']):
            data = (self.a, self.b)[index]
            self.assertEqual(item['index'], index)
            self.assertEqual(item['status'], 200)
            self.assertEqual(item['source_sha256'], hashlib.sha256(data).hexdigest())
            self.assertEqual(item['result']['source_sha256'], item['source_sha256'])
            self.assertEqual(item['result']['systems'], [dict(top=1.25, bottom=39.75)])
            self.assertEqual(item['result']['notehead_boxes'], [dict(cx=17.125, cy=22.625)])
        self.assertEqual(len({x['result']['worker_id'] for x in result['pages']}), 1)
        self.assertTrue(result['pages'][0]['result']['service_timing']['first_call'])
        self.assertFalse(result['pages'][1]['result']['service_timing']['first_call'])
        self.assertGreaterEqual(result['pages'][1]['batch_offset_seconds'], result['pages'][0]['page_seconds'])

    def test_identical_images_with_different_page_ids_remain_distinct(self):
        value = batch.detect(self.detector, AUTH, envelope(page('a', self.a), page('b', self.a)))
        self.assertEqual([x['id'] for x in value['pages']], ['a', 'b'])
        self.assertEqual(self.seen, [self.a, self.a])

    def test_authorize_before_parsing_and_no_detector_call(self):
        with self.assertRaises(RequestError) as error:
            batch.detect(self.detector, 'invalid', b'not json')
        self.assertEqual(error.exception.status, 401)
        self.assertEqual(self.seen, [])

    def test_invalid_envelopes_rejected_before_any_inference(self):
        valid = page('a', self.a)
        cases = [b'bad', b'[]', b'{"version":true,"pages":[]}', envelope(),
                 envelope(valid, valid), envelope({**valid, 'id': '../other'}),
                 envelope({**valid, 'png': '$invalid'}),
                 envelope({**valid, 'sha256': 'bad'}),
                 envelope({**valid, 'extra': True}),
                 envelope(*(page(str(n), self.a) for n in range(6)))]
        for body in cases:
            with self.subTest(body=body[:60]), self.assertRaises(RequestError) as error:
                batch.detect(self.detector, AUTH, body)
            self.assertEqual(error.exception.status, 400)
        self.assertEqual(self.seen, [])

    def test_total_byte_limit_applies_to_complete_batch(self):
        with patch.object(batch, 'MAX_BYTES', len(self.a) + len(self.b) - 1):
            with self.assertRaises(RequestError) as error:
                batch.detect(self.detector, AUTH, envelope(page('a', self.a), page('b', self.b)))
        self.assertEqual(error.exception.status, 413)
        self.assertEqual(self.seen, [])

    def test_body_limit_before_json_allocation(self):
        with patch.object(batch, 'MAX_BODY', 1), self.assertRaises(RequestError) as error:
            batch.detect(self.detector, AUTH, b'{}')
        self.assertEqual(error.exception.status, 413)

    def test_bad_png_and_hash_are_page_errors_without_discarding_good_page(self):
        body = envelope(page('good', self.a), page('bad-png', b'invalid'),
                        {**page('bad-hash', self.b), 'sha256': '0' * 64})
        result = batch.detect(self.detector, AUTH, body)
        self.assertEqual(result['status'], 'partial')
        self.assertEqual([x['status'] for x in result['pages']], [200, 400, 400])
        self.assertEqual(self.seen, [self.a])

    def test_unsupported_page_preserves_good_geometry_and_continues(self):
        original = self.detector.detector
        def detect(path, model_path):
            if path.read_bytes() == self.b:
                raise ValueError('unsupported')
            return original(path, model_path)
        self.detector.detector = detect
        result = batch.detect(self.detector, AUTH, envelope(page('bad', self.b), page('good', self.a)))
        self.assertEqual(result['status'], 'partial')
        self.assertEqual([x['status'] for x in result['pages']], [422, 200])
        self.assertEqual(result['pages'][0]['error'], 'no_supported_geometry')
        self.assertNotIn('unsupported', json.dumps(result))

    def test_http_batch_negotiated_compression_and_source_offer(self):
        original = self.detector.detector
        def large(path, model_path):
            result = original(path, model_path)
            result['systems'] *= 500
            return result
        self.detector.detector = large
        for accept in ('', 'gzip'):
            request = func.HttpRequest(method='POST', url='https://example.test/api/detect-batch',
                headers={'Authorization': AUTH, 'Accept-Encoding': accept},
                body=envelope(page('one', self.a), page('two', self.b)))
            with patch.object(http, 'service', return_value=self.detector), \
                    patch.object(http, 'source_url', return_value='https://example.test/source'):
                response = http.detect_batch.build().get_user_function()(request)
            self.assertEqual(response.status_code, 200)
            body = response.get_body()
            if accept:
                self.assertEqual(response.headers['Content-Encoding'], 'gzip')
                body = gzip.decompress(body)
            else:
                self.assertNotIn('Content-Encoding', response.headers)
            value = json.loads(body)
            self.assertEqual(value['source_url'], 'https://example.test/source')
            self.assertEqual(value['model_sha256'], http.MODEL_SHA256)
            self.assertRegex(value['service_timing']['instance_id'], r'^[0-9a-f]{16}$')
            self.assertEqual([x['id'] for x in value['pages']], ['one', 'two'])
            self.assertEqual(value['status'], 'complete')

    def test_http_envelope_error_is_structured_and_does_not_infer(self):
        request = func.HttpRequest(method='POST', url='https://example.test/api/detect-batch',
                                  headers={'Authorization': 'bad'}, body=b'bad')
        with patch.object(http, 'service', return_value=self.detector):
            response = http.detect_batch.build().get_user_function()(request)
        self.assertEqual(response.status_code, 401)
        self.assertEqual(json.loads(response.get_body()), {'error': 'unauthorized'})
        self.assertEqual(self.seen, [])


if __name__ == '__main__':
    unittest.main()

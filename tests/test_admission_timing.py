import gzip
import json
import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import azure.functions as func
import function_app as http


class AdmissionTimingTests(unittest.TestCase):
    def test_timing_preserves_geometry_and_does_not_mutate_detector_result(self):
        original = {'systems': [{'top': 1.25, 'bottom': 98.75}] * 500,
                    'worker_id': 'process-1', 'service_timing': {'detector_seconds': 2.5}}
        detector = SimpleNamespace(detect=lambda *args: original)
        for accept in ('', 'gzip'):
            request = func.HttpRequest(method='POST', url='https://example.test/api/detect',
                                       headers={'Accept-Encoding': accept}, body=b'exact-png')
            before = time.time()
            with patch.object(http, 'service', return_value=detector):
                response = http.detect.build().get_user_function()(request)
            data = response.get_body()
            if accept:
                self.assertEqual(response.headers['Content-Encoding'], 'gzip')
                data = gzip.decompress(data)
            result = json.loads(data)
            timing = result.pop('service_timing')
            self.assertLessEqual(before, timing['http_entry_at'])
            self.assertLessEqual(timing['http_entry_at'], time.time())
            self.assertGreaterEqual(timing['factory_seconds'], 0)
            self.assertGreaterEqual(timing['body_seconds'], 0)
            self.assertEqual(timing['detector_seconds'], 2.5)
            self.assertRegex(timing['instance_id'], r'^[0-9a-f]{16}$')
            self.assertEqual(result, {k:v for k,v in original.items() if k != 'service_timing'})
            self.assertEqual(original['service_timing'], {'detector_seconds': 2.5})

    def test_health_identifies_ready_process_without_loading_model_or_exposing_host_name(self):
        detector = SimpleNamespace(model_ready=True, worker_id='ready-process')
        request = func.HttpRequest(method='GET', url='https://example.test/api/health', body=b'')
        with patch.object(http, '_service', detector), patch.object(http, 'service') as factory, \
                patch.dict(os.environ, {'WEBSITE_INSTANCE_ID':'private-host-name'}):
            response = http.health.build().get_user_function()(request)
            factory.assert_not_called()
        value = json.loads(response.get_body())
        self.assertTrue(value['model_ready'])
        self.assertEqual(value['worker_id'], 'ready-process')
        self.assertNotIn(b'private-host-name', response.get_body())
        self.assertRegex(value['instance_id'], r'^[0-9a-f]{16}$')


if __name__ == '__main__':
    unittest.main()

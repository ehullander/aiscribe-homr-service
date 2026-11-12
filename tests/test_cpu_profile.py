import base64
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import azure.functions as func
from PIL import Image
import function_app as http
from service import StructuralService
from structural import batch
from structural.cpu_profile import CPUProbe


class CPUProfileTests(unittest.TestCase):
    def test_v2_cpu_work_throttle_deltas_and_two_core_quota(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            stat = root/'cpu.stat'
            stat.write_text('usage_usec 1000000\nthrottled_usec 500000\nnr_periods 10\nnr_throttled 2\n')
            (root/'cpu.max').write_text('200000 100000')
            (root/'cpu.stat.local').write_text('throttled_usec 200000')
            wall = iter([10.0, 12.0]); cpu = iter([3.0, 6.0])
            probe = CPUProbe(root, wall=lambda:next(wall), cpu=lambda:next(cpu)).start()
            stat.write_text('usage_usec 5000000\nthrottled_usec 1500000\nnr_periods 30\nnr_throttled 12\n')
            (root/'cpu.stat.local').write_text('throttled_usec 600000')
            result = probe.finish()
        self.assertEqual(result['wall_seconds'], 2)
        self.assertEqual(result['process_cpu_seconds'], 3)
        self.assertEqual(result['process_cpu_to_wall'], 1.5)
        self.assertEqual(result['cpu_quota_cores'], 2)
        self.assertEqual(result['cgroup_usage_seconds'], 4)
        self.assertEqual(result['cgroup_throttled_seconds'], 1)
        self.assertEqual(result['cgroup_local_throttled_seconds'], .4)
        self.assertEqual(result['cgroup_periods'], 20)
        self.assertEqual(result['cgroup_throttled_periods'], 10)

    def test_unavailable_or_reset_counters_are_unknown_not_fabricated_zero(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            probe = CPUProbe(root).start()
            result = probe.finish()
            self.assertFalse(result['cgroup_stats_available'])
            self.assertIsNone(result['cgroup_throttled_seconds'])
            self.assertIsNone(result['cpu_quota_cores'])
            (root/'cpu.stat').write_text('usage_usec 100\n')
            probe = CPUProbe(root).start()
            (root/'cpu.stat').write_text('usage_usec 10\n')
            self.assertIsNone(probe.finish()['cgroup_usage_seconds'])

    def test_v1_units_and_quota_are_normalized(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); folder = root/'cpu,cpuacct';folder.mkdir()
            (folder/'cpu.stat').write_text('throttled_time 1000000000\nnr_periods 10\nnr_throttled 2\n')
            (folder/'cpuacct.usage').write_text('1000000000')
            (folder/'cpu.cfs_quota_us').write_text('200000')
            (folder/'cpu.cfs_period_us').write_text('100000')
            probe = CPUProbe(root).start()
            (folder/'cpu.stat').write_text('throttled_time 2000000000\nnr_periods 20\nnr_throttled 3\n')
            (folder/'cpuacct.usage').write_text('3000000000')
            result = probe.finish()
            self.assertEqual(result['cgroup_throttled_seconds'], 1)
            self.assertEqual(result['cgroup_usage_seconds'], 2)
            self.assertEqual(result['cpu_quota_cores'], 2)

    def detector(self):
        return StructuralService('x'*40, 'unused', detector=lambda *a,**kw:dict(systems=[{'top':1.25}]))

    def png(self):
        out=BytesIO();Image.new('L',(20,30),255).save(out,format='PNG');return out.getvalue()

    def test_default_calls_do_no_diagnostic_reads_or_change_geometry(self):
        detector = self.detector()
        with patch('structural.cpu_profile.CPUProbe') as probe:
            result = detector.detect('Bearer '+'x'*40, self.png())
        probe.assert_not_called()
        self.assertNotIn('cpu_profile',result['service_timing'])
        self.assertEqual(result['systems'],[{'top':1.25}])

    def test_profile_failure_cannot_turn_successful_detection_into_error(self):
        detector=self.detector()
        for phase in ['start','finish']:
            with self.subTest(phase=phase), patch('structural.cpu_profile.CPUProbe') as cls:
                obj=cls.return_value
                obj.start.return_value=obj
                getattr(obj,phase).side_effect=OSError('inaccessible system diagnostics')
                result=detector.detect('Bearer '+'x'*40,self.png(),profile_cpu=True)
                self.assertEqual(result['systems'],[{'top':1.25}])
                self.assertEqual(result['service_timing']['cpu_profile'],{'available':False})

    def test_http_profile_header_is_opt_in_and_batch_uses_same_flag(self):
        calls=[]
        def detect(*args,**kwargs):
            calls.append(kwargs)
            return dict(systems=[],service_timing={})
        detector=SimpleNamespace(detect=detect)
        for flag in ['', '1']:
            req=func.HttpRequest(method='POST',url='https://example.test/api/detect',
                headers={'X-AIScribe-Profile-CPU':flag},body=b'unchanged')
            with patch.object(http,'service',return_value=detector):
                http.detect.build().get_user_function()(req)
        self.assertEqual(calls,[{}, {'profile_cpu':True}])
        detector=self.detector();data=self.png()
        body=json.dumps({'version':1,'pages':[dict(id='one',sha256=__import__('hashlib').sha256(data).hexdigest(),png=base64.b64encode(data).decode())]}).encode()
        value=batch.detect(detector,'Bearer '+'x'*40,body,profile_cpu=True)
        self.assertIn('cpu_profile',value['pages'][0]['result']['service_timing'])
        self.assertEqual(value['pages'][0]['result']['systems'],[{'top':1.25}])


if __name__=='__main__':unittest.main()

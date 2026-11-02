"""Isolated CPU experiment, never installed in the production HOMR app.

SPDX-License-Identifier: AGPL-3.0-or-later
"""
import ast
from contextlib import ExitStack
import gc
import hashlib
import inspect
import platform
import resource
from textwrap import dedent
from threading import Lock
from time import perf_counter, process_time
from unittest.mock import patch
from uuid import uuid4

from service import StructuralService, RequestError

THREADS = (1, 2, 4)
BATCHES = (1, 2, 4, 8)
MASKS = ('notehead', 'symbols', 'staff', 'clefs_keys', 'stems_rest')


def batch_variant(function, size):
    tree = ast.parse(dedent(inspect.getsource(function)))
    changed = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Call)
                and isinstance(node.left.func, ast.Name) and node.left.func.id == 'len'
                and len(node.left.args) == 1 and isinstance(node.left.args[0], ast.Name)
                and node.left.args[0].id == 'tiles' and len(node.ops) == 1
                and isinstance(node.ops[0], ast.Eq) and len(node.comparators) == 1
                and isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value == 2):
            node.comparators[0].value = size
            changed += 1
    if changed != 1:
        raise ValueError('The pinned predictor changed; review before benchmarking')
    namespace = dict(function.__globals__)
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(function), 'exec'), namespace)
    return namespace[function.__name__]


class RuntimeBenchmark:
    def __init__(self, token, model, session_factory=None):
        self.token, self.model = token, model
        self.session_factory = session_factory
        self.lock = Lock()
        self.session = None
        self.setting = None
        self.worker_id = uuid4().hex
        # Reuse production auth, PNG/size/hash validation and scratch cleanup.
        # Only a model session is reused; images and results are never cached.
        StructuralService(token, model, detector=lambda *a, **k: {})

    def run(self, authorization, data, expected_hash, threads, batch):
        if threads not in THREADS or batch not in BATCHES:
            raise RequestError(400, 'invalid_benchmark_setting')
        record = {}
        def detect(path, model_path):
            from structural import detector, runtime
            import numpy as np
            import cv2
            import onnxruntime as ort
            load_seconds = 0.0
            if self.setting != (threads, batch):
                self.session = None
                gc.collect()
                started = perf_counter()
                if self.session_factory:
                    self.session = self.session_factory(model_path, threads)
                else:
                    options = ort.SessionOptions()
                    options.intra_op_num_threads = threads
                    options.inter_op_num_threads = 1
                    self.session = ort.InferenceSession(str(detector._verify_model(model_path)),
                        sess_options=options, providers=['CPUExecutionProvider'])
                self.session.run(['output'], {'input': np.full((batch, 3, 320, 320), 255, np.float32)})
                self.setting = (threads, batch)
                load_seconds = perf_counter() - started
            original = detector._predict
            variant = batch_variant(original, batch)
            samples = []
            owner = self
            class MeasuredSession:
                def run(self, outputs, inputs):
                    started = perf_counter()
                    result = owner.session.run(outputs, inputs)
                    samples.append(dict(batch=int(inputs['input'].shape[0]), seconds=perf_counter()-started))
                    return result
            def predict(image, model):
                started, cpu_started = perf_counter(), process_time()
                prediction = variant(image, model)
                record.update(prediction_seconds=perf_counter()-started,
                    prediction_cpu_seconds=process_time()-cpu_started,
                    prepared_shape=list(image.shape),
                    prepared_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                    masks={key: dict(shape=list(getattr(prediction, key).shape),
                        sha256=hashlib.sha256(getattr(prediction, key).tobytes()).hexdigest()) for key in MASKS})
                return prediction
            started, cpu_started = perf_counter(), process_time()
            with ExitStack() as stack:
                stack.enter_context(patch.object(runtime, 'session_for', lambda model: MeasuredSession()))
                stack.enter_context(patch.object(detector, '_predict', predict))
                result = detector.detect(path, model_path=model_path)
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            record.update(detector_seconds=perf_counter()-started,
                detector_cpu_seconds=process_time()-cpu_started,
                session_load_and_warmup_seconds=load_seconds,
                inference_seconds=sum(s['seconds'] for s in samples), batches=samples,
                peak_process_rss_bytes=peak if platform.system()=='Darwin' else peak*1024,
                platform=platform.platform(), machine=platform.machine(),
                versions=dict(onnxruntime=ort.__version__, numpy=np.__version__, opencv=cv2.__version__),
                predictor_sha256=hashlib.sha256(inspect.getsource(original).encode()).hexdigest())
            from pathlib import Path
            record['container_limits'] = {}
            for name in ('cpu.max', 'memory.max'):
                limit = Path('/sys/fs/cgroup') / name
                if limit.is_file():
                    record['container_limits'][name] = limit.read_text().strip()
            return result
        with self.lock:
            # All setting changes and monkeypatches are isolated to this test
            # process and serialized. This app has no production detect route.
            service = StructuralService(self.token, self.model, detector=detect)
            geometry = service.detect(authorization, data, expected_hash)
        return dict(worker_id=self.worker_id, threads=threads, batch=batch,
                    measurement=record, geometry=geometry)

"""Reuse verified CPU sessions; do not load a model for each crop.

SPDX-License-Identifier: AGPL-3.0-or-later
"""
from functools import lru_cache


@lru_cache(maxsize=1)
def session_for(model):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])

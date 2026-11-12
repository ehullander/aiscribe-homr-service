"""Bounded page batch; retain normal per-page detection. AGPL-3.0-or-later."""
import base64
import binascii
from hmac import compare_digest
import json
import re
from time import perf_counter

from service import MAX_BYTES, RequestError

MAX_PAGES = 5
MAX_BODY = ((MAX_BYTES + 2) // 3) * 4 + 8192
PAGE_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}')
SHA256 = re.compile(r'[0-9a-f]{64}')


def pages(authorization, body, token):
    # Authorize before parsing/allocating image payloads. Envelope validation
    # completes before inference; PNG/hash validation stays in normal detect.
    if not compare_digest(authorization or '', 'Bearer ' + token):
        raise RequestError(401, 'unauthorized')
    if not body or len(body) > MAX_BODY:
        raise RequestError(413, 'invalid_batch_size')
    try:
        value = json.loads(body)
        if (not isinstance(value, dict) or set(value) != {'version', 'pages'}
                or type(value['version']) is not int or value['version'] != 1
                or not isinstance(value['pages'], list)
                or not 1 <= len(value['pages']) <= MAX_PAGES):
            raise ValueError()
        seen = set()
        result = []
        total = 0
        for item in value['pages']:
            if (not isinstance(item, dict) or set(item) != {'id', 'png', 'sha256'}
                    or not isinstance(item['id'], str) or not PAGE_ID.fullmatch(item['id'])
                    or item['id'] in seen or not isinstance(item['sha256'], str)
                    or not SHA256.fullmatch(item['sha256']) or not isinstance(item['png'], str)):
                raise ValueError()
            seen.add(item['id'])
            data = base64.b64decode(item['png'], validate=True)
            if not data or len(data) > MAX_BYTES:
                raise RequestError(413, 'invalid_image_size')
            total += len(data)
            if total > MAX_BYTES:
                raise RequestError(413, 'invalid_batch_size')
            result.append((item['id'], data, item['sha256']))
        return result
    except (ValueError, TypeError, KeyError, binascii.Error):
        raise RequestError(400, 'invalid_batch') from None


def detect(detector, authorization, body):
    started = perf_counter()
    inputs = pages(authorization, body, detector.token)
    decoded = perf_counter()
    results = []
    for index, (identity, data, digest) in enumerate(inputs):
        page_started = perf_counter()
        row = dict(id=identity, source_sha256=digest,
                   batch_offset_seconds=page_started-started)
        try:
            # Each original PNG follows the same validation, preprocessing,
            # model session, geometry, and source coordinates as /detect.
            row.update(status=200, result=detector.detect(authorization, data, digest))
        except RequestError as error:
            row.update(status=error.status, error=error.code)
        except ValueError:
            row.update(status=422, error='no_supported_geometry')
        row['page_seconds'] = perf_counter()-page_started
        row['index'] = index
        results.append(row)
    return dict(version=1, pages=results,
                status='complete' if all(x['status'] == 200 for x in results) else 'partial',
                batch_seconds=perf_counter()-started,
                envelope_seconds=decoded-started)

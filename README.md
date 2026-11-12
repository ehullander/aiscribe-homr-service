# HOMR structural CPU service

A separate AGPL-3.0-or-later service using the HOMR 0.7.0 segmentation model and
staff/note/bar geometry. No chord recognizer, MusicXML transformer, OpenAI key,
SQL connection, user database or Apple runtime is included.

`provenance.json` records upstream files and every extracted file hash. The
structural wrapper is copied from AIScribe's accepted detector. The only runtime
changes disable Apple-native imports, reuse one verified CPU ONNX session, and
remove the inherited geometry-result cache. Every request performs inference.
Image preparation, model input, tiling, thresholds and geometry are unchanged.

Install `requirements.txt` in Python 3.13. Put the exact segmentation weight at
`models/segnet.onnx`, SHA-256
`6ed36640db4ef5d223098b6d5efe4eda97c66b24a2c72faab8a018c749003a8d`.
It is the HOMR 0.7.0 `segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f.onnx`
weight. The deployment extractor verifies the original bundled artifact.

Set `HOMR_SERVICE_TOKEN` to a dedicated random credential (at least 32 characters)
and `HOMR_SOURCE_URL` to the exact published source commit for this deployment.
Deploy with Azure Functions using a CPU Flex Consumption plan. GPU providers are
not enabled. POST the original PNG bytes to `/api/detect` using Bearer auth and
optional `X-Source-SHA256`. This upload is neither recompressed nor pre-resized by
the transport. Existing HOMR preprocessing is performed inside the detector.

`/api/source` publishes the corresponding-source link. `/api/health` identifies
the model without triggering inference. Source requests and inputs never enter
SQL. Inference scratch files are removed after each request. Only the verified model session is retained per process. Chart geometry and
image results are not cached across requests. One CPU inference executes
at a time per instance; the hosting plan can scale instances independently.

The surrounding app communicates through this service boundary. This repository
contains its corresponding source and license; repository separation alone is
not a legal determination about any larger deployment.
# Admission diagnostics

The HTTP adapter adds entry time, module load time, an opaque machine hash,
service-factory time and body-read time under `service_timing`. The existing
detector geometry, model, serial inference lock and input PNG remain unchanged.
Health also identifies its prepared process and machine without starting an
inference. These fields distinguish Azure admission from actual detector work;
cross-machine clock differences must be considered when comparing timestamps.
No chart, credential or additional persistent diagnostic is logged.


# Isolated HOMR runtime grid

This temporary service benchmarks the pinned HOMR detector on the same 4 GiB
Flex Consumption CPU tier as the live service. It is never deployed over the
live HOMR app. No OpenAI key, beta credential map, SQL access or user storage
connection is supplied. The controller sends the frozen ten-chart source PNGs.

One deployment supports intra-op threads 1, 2, 4 and tile batches 1, 2, 4, 8.
The current 2-thread/2-tile setting is the control. CPUExecutionProvider,
sequential graph execution, model weights, image preparation, class mapping,
geometry rules and all prompts remain fixed. The benchmark-only AST transform
changes exactly the existing `len(tiles) == 2` flush threshold and rejects source
that no longer contains that single expected expression.

Initial grid: 12 settings × 17 actual inputs from 10 charts = 204 fresh requests.
Seven identical-image reuses from the original pipeline are not new inferences.
After the grid, confirm the current default and up to two promising settings with
two additional rounds, reversing order. Do not adopt a setting based on one run.
The controller is serial so it measures request latency without imposed load.
This is not a peak-load test. It stops on a failed or ambiguous request and keeps
the dispatch, source identity and error; it never blindly replays that dispatch.

`POST /api/benchmark` accepts PNG bytes with Bearer authorization and headers
`X-Source-SHA256`, `X-Benchmark-Threads`, `X-Benchmark-Batch`. It returns complete
geometry plus hashes of all five neural masks before postprocessing, prepared
pixels, wall/CPU times, every inference batch, worker identity and Linux resource
limits. Model loading/warmup is timed separately. Images and predictions are not
cached across requests. The current model session is reused and replaced when
the tested setting changes. Results live in the caller's evidence directory.

The public source offer must include these benchmark modules, the exact pinned
base and license before deployment. Package with
`tools/azure/package_homr_benchmark.py`. Infrastructure is declared in
`infra/homr-benchmark.bicep`. Use a new, dedicated resource group and random private
credential. Retain evidence, then remove the whole temporary resource group;
do not leave a warm test worker running. The live app and its settings remain
unchanged. This is CPU only; no GPU deployment is authorized here.

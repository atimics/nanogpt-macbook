# Local validation

Measured on 28 September 2026 with an Apple M4 Max, 36 GiB unified memory,
macOS 26.3.1, native Python 3.12.13, and MLX 0.32.3.

## Metal training

Command:

```bash
nanogpt prepare --demo --out data/demo
nanogpt train --data data/demo --run runs/demo-gpu --steps 300 \
  --eval-every 100 --eval-batches 10 --log-every 50
```

- Tiny model: 837,888 parameters, context 128, batch 8, float32.
- Data: bundled 4,544-byte story; 4,089 training bytes and 455 validation bytes.
- Source SHA-256: `4a03023faf87edfc1e9b66cc609d85f38ff104fcce444faac3050a5c66792a42`.
- Initial validation loss: 5.5229.
- Step 300 training loss: 1.9995; validation loss: 2.1712 (3.132 bits per byte).
- Warm log windows: about 62,000 to 66,000 byte tokens per second.
- Peak MLX allocation: about 189 MiB.
- Resume to step 310 completed; validation loss reached 2.1683.

These figures describe one small demo run on this machine. A larger corpus,
another model preset, or a different Mac will produce different results. The
short story is a plumbing and learning check. Its held-out segment provides a
small validation sample. Text quality needs further training and broader data.

## Automated checks

The suite passed all 26 tests using MLX on the CPU. It covers:

- Causal attention: future input tokens preserve earlier logits.
- Learning: training reduces held-out loss on a repeatable text fixture.
- Resume: interrupted and continuous runs agree on model and optimizer state.
- Random state: checkpoint restore reproduces the next MLX and NumPy draws.
- Gradient accumulation: small batches agree with one larger batch.
- Data: UTF-8 round trips, split boundaries, shifted targets, and hash checks.
- Recovery: run locks, failed checkpoint writes, time limits, and Ctrl+C.
- CLI: prepare, train, resume, sample, evaluate, and input errors.

Ruff lint and format checks pass. GitHub CI runs the same suite on Linux with
the MLX CPU backend. The PR check records the result for each commit.

The source distribution and wheel build successfully. The wheel was installed
in a fresh environment; its command entry point and bundled demo both work.

## Metal speed update, 29 September 2026

Source commit `17cb28a` uses fused causal softmax kernels and a 2 GiB buffer
cache. A fresh baseline at `0eb0273` used the same Mac, dependencies, float32
models, batch sizes, 20 warmup steps, and three trials of 100 timed steps.

| Preset / device | Baseline bytes/s | Updated bytes/s | Median change |
| --- | ---: | ---: | ---: |
| tiny / Metal | 308,716 | 301,597 | -2.3% |
| small / Metal | 111,185 | 122,751 | +10.4% |
| medium / Metal | 41,609 | 45,797 | +10.1% |
| tiny / CPU | 23,737 | 23,292 | -1.9% |

The small and medium trial ranges are separated from their baseline ranges.
Tiny's ranges overlap. Thermal state and other apps affect timing. The earlier
recordings had lower throughput, so this table uses the fresh baseline to
measure the code change. Every trial is in [`benchmarks/results`](../benchmarks/results/).

All 55 local tests pass with `PYTHONPATH=src`, including CPU coverage, Metal
output and gradient comparisons, odd sequence lengths, the causal boundary,
large attention scores, and GPU resume with gradient accumulation. Ruff,
format checks, the static page build, the wheel, and the source archive pass.
Linux CI runs the CPU checks and skips the Metal checks.

The updated tiny model also completed 300 steps on the bundled story. Validation
loss fell from 5.5229 to 2.1712. Its [learning receipt](../benchmarks/learning/demo-17cb28a.json)
records each reported training and validation point.

## GELU gradient update, 29 September 2026

Source commit `f635c16` uses an explicit stable GELU derivative on Metal.
The reference forward activation, float32 settings, optimizer, and benchmark
protocol remain the same. The CPU uses MLX's native derivative. A fresh
baseline at `5802058` and the updated source ran in the same session.

| Preset / device | Baseline bytes/s | Updated bytes/s | Median change | Peak memory before / after |
| --- | ---: | ---: | ---: | ---: |
| tiny / Metal | 316,312 | 319,800 | +1.1% | 167.4 / 149.4 MiB |
| small / Metal | 127,417 | 134,523 | +5.6% | 1113.0 / 1033.1 MiB |
| medium / Metal | 46,068 | 48,703 | +5.7% | 2054.5 / 2033.1 MiB |
| tiny / CPU | 23,700 | 23,617 | -0.4% | 109.3 / 109.3 MiB |

Small's GPU trial ranges are separated. Tiny and medium have overlapping
ranges. Timing varies with other apps and machine state. All raw timings and
source hashes are published in `benchmarks/results/`. The largest active-memory
reduction is 10.8%, for tiny on Metal.

All 63 local tests pass. The added checks compare compiled and eager gradients
with MLX on CPU and GPU, check strided batches, compare the derivative with
float64 finite differences, cover saturated values, and compare every gradient
in a two-layer GPT. The suite also checks checkpoint resume with accumulation.
Ruff lint and formatting pass.

The 300-step story run reached validation loss 2.1712, matching the previous
result to four decimal places. Its [learning receipt](../benchmarks/learning/demo-f635c16.json)
records the source hash and reported metrics.

## Benchmark layout update, 29 September 2026

At a 320-pixel viewport, the prior detailed-results table was 701 pixels wide
inside a 276-pixel container. It required sideways scrolling. The expanded
results now use labeled cards on narrow screens. The complete table stays
available on wide screens. A disclosure keeps the chart overview short.

Fresh timeline plots already fit at the tested widths. Returning browsers
could reuse layout assets: GitHub Pages serves them with a 600-second cache,
and earlier builds reused the same URLs. This is a possible cause of the
reported chart behavior. The builder now gives changed CSS and JavaScript
new content-based filenames. An integration test verifies those URLs and files.

Plots now have explicit width bounds, inward endpoint labels, and label spacing
based on rendered text. Browser checks at 320, 375, 600, 768, 851, 960, 1024,
1100, 1101, and 1440 pixels showed matching container and scroll widths for the
plots and expanded results. All 24 measured points and both comparison
percentages remain. A temporary 60-point-per-series layout test also fits at
320 pixels with all links present. CPU filtering, memory selection, and
keyboard expansion of the results passed. The eight site tests, Ruff, and
JavaScript syntax check pass.

## LayerNorm gradient update, 30 September 2026

Source `b3380d8` reduces weight and bias gradient buffers before writing them
to device memory. Four rows share each Metal threadgroup. The forward operation,
float32 weights, model configuration, optimizer, and checkpoint layout stay the
same. Source `2c34805` adds a repeatable comparison tool and shares the same
training-source hash.

All 99 local tests pass. The new cases compare eager and compiled values and all
three gradients with native MLX, cover strided arrays and odd row counts, check
float64 finite differences with constant and offset inputs, and compare every
gradient in two-layer GPTs at widths 32, 128, 256, and 384. CPU, half precision,
wide inputs, and optional affine parameters exercise the native path. Existing
tests cover learning, checkpoint resume, and gradient accumulation.

Standard receipts at `9060ccc` and `b3380d8` record these GPU medians:

| Preset | Baseline bytes/s | Updated bytes/s | Peak memory before / after |
| --- | ---: | ---: | ---: |
| tiny | 295,575 | 246,194 | 149.4 / 146.9 MiB |
| small | 82,462 | 84,718 | 1033.1 / 1011.0 MiB |
| medium | 33,115 | 30,315 | 2033.1 / 1907.5 MiB |

The CPU control changed from 17,881 to 16,474 bytes/s with the same native CPU
math. Its peak stayed at 109.3 MiB. Machine timing varied enough to change the
apparent direction of the GPU comparison. All standard receipts remain in the
published time series.

An additional comparison at `2c34805` alternates native and grouped paths in 200
pairs of complete single steps. Median gains were +4.2% for tiny, -0.4% for small,
and +3.3% for medium. Medium improved in 181 of 200 pairs. The total-time changes
were +1.4%, -1.2%, and +3.5%. This diagnostic uses two live models and a 4 GiB
memory/cache limit. The standard memory measurements use one model and 2 GiB.
The [protocol and raw receipts](../benchmarks/README.md#layernorm-comparison)
include both the adjacent-step and earlier block comparisons.

The 300-step story run reached validation loss 2.1712157, matching the previous
2.1712109 to four decimal places. Its [learning receipt](../benchmarks/learning/demo-2c34805.json)
contains all reported metrics and the source hash.

## Gradient clipping update, 30 September 2026

Source `17ba7ea` groups float32 gradient reductions on Metal. It preserves the
global L2 norm, clipping scale, epsilon, and AdamW update. All 124 local tests
pass. Coverage includes CPU and Metal, eager and compiled calls, scalar and
empty leaves, strided and broadcast arrays, large reductions, non-finite
values, and repeated full optimizer updates with gradient accumulation.

The standard benchmark compares `fa60b54` with `17ba7ea`. Median throughput
rose 1.0% for tiny, 1.9% for small, and 0.3% for medium. Tiny and medium have
overlapping trial ranges. In 200 adjacent pairs of complete training steps,
median throughput rose 3.5%, 1.4%, and 2.6%, respectively. The updated path
was faster in 188, 186, and 193 pairs. This diagnostic uses two live models
and a 4 GiB memory/cache limit; the time series uses one model and 2 GiB.

Peak active memory changed from 146.9 to 144.2 MiB for tiny, 1011.0 to
1033.0 MiB for small, and 1907.5 to 2034.9 MiB for medium. This is a speed
and memory tradeoff for the larger presets. The [protocol and raw receipts](../benchmarks/README.md#gradient-clipping-comparison)
record both timing methods and the measured memory.

The 300-step story run reached validation loss 2.1712059, matching the prior
2.1712157 to four decimal places. The [learning receipt](../benchmarks/learning/demo-17ba7ea.json)
contains the source hash and each reported metric.

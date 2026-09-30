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

## Chart label bounds, 29 September 2026

The current 38-point, 11-commit history fits in Chromium and WebKit at widths
from 280 to 1440 pixels. Rendered glyph checks also cover the chart content.

Two stress cases exposed label failures. With later GPU-only measurements,
the older CPU series stays near the left of the shared commit axis. Its
forced first and last labels overlap. A single measurement near the right
edge skips fitting and clips both its value and commit hash.

Label fitting now reserves space for the latest value and hash. It moves
labels inside the SVG bounds and hides labels that collide. Every point keeps
its global commit position and link. Long comparison percentages wrap within
the card. Both percentage comparisons stay visible.

Chromium and WebKit checks cover the current data, sparse and single-point
histories, long percentages, repeated width changes, keyboard focus, the
expanded result cards, CPU filtering, and memory selection. Four regression
tests use Node's built-in test runner; the Pages build runs them. The eight
Python site tests, Ruff, JavaScript syntax check, and site build pass.

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

## Saved LayerNorm statistics, 30 September 2026

Source `231140f` groups the forward pass and reuses each row's statistics for
gradients. Larger backward groups also reduce partial gradient buffers. The
row origin and shifted mean are stored separately to retain accuracy when
input values are nearly equal.

All 130 local tests pass. The added output checks compare individual values
with float64 calculations for strided, constant, and nearly constant rows,
in eager and compiled calls. Existing tests cover all gradients, finite
differences, complete model gradients, optimizer updates, accumulation, and
checkpoint resume. Ruff and formatting pass.

In 200 adjacent pairs against the prior LayerNorm at `63a55cb`, median
throughput rose 4.2% for tiny, 0.6% for small, and 1.0% for medium. The updated
path was faster in 126, 124, and 168 pairs, respectively. Peak active memory
fell from 144.2 to 142.6 MiB, 1033.0 to 1022.9 MiB, and 2034.9 to 2022.7 MiB.

The standard sequential runs show larger timing differences. The published
time series retains those measured values and ranges. Use the [paired protocol
and receipts](../benchmarks/README.md#saved-layernorm-statistics) to assess this
code change; they include the previous source commit and its file hash.

The 300-step story run reached validation loss 2.1712095, matching the prior
2.1712059 to four decimal places. Its [learning receipt](../benchmarks/learning/demo-231140f.json)
contains the source hash and all reported metrics.

## Chart container and text sizing, 30 September 2026

The 44-point page fits ordinary desktop and phone viewports. Two additional
browser settings exposed overflow in Chromium and WebKit:

- With a 360 px results container inside a 1280 px viewport, bar rows needed
  254 px in 206 px of space. The timeline grid kept two columns and clipped
  its axis title.
- With Chromium's minimum font size set to 20 px, a 320 px page needed
  291 px for a 244 px memory row. Selecting Memory moved the page sideways.
  At 1280 px, the expanded result table also needed more than its 1104 px
  container.

Timeline columns now follow their container width. Labels and values share
natural space above full-width bars. Controls, the header, and phone highlights
wrap with larger text. The result table measures its rendered width and uses
cards when needed. Opening the details, changing the device filter, or resizing
the container runs that check. Height changes retain the chosen layout.

The browser audit covers Chromium, Chromium with 20 px minimum text, and
WebKit at widths 240, 280, 320, 600, 851, and 1280 px. It also sets
`#results { max-width: 360px }` at 1280 px. Both chart metrics fit in all 42
states: document width equals viewport width, horizontal scroll position is
zero, and visible SVG text and points stay within the plot. The narrow
container's bar rows now use 206 px; the 320 px memory rows use 244 px.

A second audit opens the result details with the keyboard, changes each device
filter, and cycles between 320 and 1280 px. It checks table and card widths,
stable layout, focus, and browser errors. A 224-point fixture adds 60 GPU-only
commits to the shared axis to check a long history with sparse CPU data.
The real page retains all 44 points and all eight baseline/previous badges.
Six Node regressions and eight Python site tests pass, as do Ruff and format
checks.

To repeat the large-text check in Chromium, launch a test browser with
`--blink-settings=minimumFontSize=20`, open the built page at 320 px, and select
Memory. For the container check, apply the rule above at 1280 px. Open
“All measured results and settings”, change each device filter, and resize down
and up. Check the whole page, visible axis text, values, and expanded results.

## Fused attention score gradients, 30 September 2026

The new Metal path uses masked matrix products and combines the probability
gradient product with the softmax derivative. It covers the small and medium
presets. The tiny preset keeps its existing attention path.

Checks cover eager and compiled outputs, all input gradients, strided inputs,
partial 64-row mask blocks, a float64 reference, causal boundaries, large
scores, complete model updates, gradient accumulation, and checkpoint resume.
The float64 gradient checks cover contexts 256 and 512 with random and uniform
scores. General attention remains covered for other widths, lengths, value
widths, and float16 inputs.

The standard three-trial receipts at `0898920` and `ddb8a6f` use the same
settings. The 200-pair comparison shows median throughput gains of 9.1% for
small and 6.0% for medium. Peak active memory falls by 47.6 and 140.4 MiB.
Tiny's measured pair difference is -0.7%, with the same attention operations.

Three small-model learning comparisons reach final validation losses between
2.2186 and 2.2997 for the baseline, and 2.2468 and 2.2999 for the update. The
largest final difference is +0.0394 nats. The saved best-checkpoint losses
differ by -0.0106, +0.000004, and +0.0007 nats for seeds 1337, 17, and 42.
Full traces, raw timings, and repeat
commands are in the [attention report](../benchmarks/README.md#attention-score-gradients).
The tiny story check reaches 2.1712084 after 300 steps.

## Attention tensor layout check, 30 September 2026

Profiling exposed a layout failure in the pinned MLX masked-matrix backend.
Sliced and reversed operands can be copied internally while the product
keeps their earlier batch strides. Source `974dde6` explicitly packs Q/K/V
and the incoming gradient before those products.

A new sliced-input regression fails on the prior source. All 185 local tests
pass with the fix, including 16 eager/compiled layout cases and the existing
float64 gradient, model-update, accumulation, and resume checks. The six Node
chart checks, site build, Ruff, and formatting pass.

The [layout report](../benchmarks/README.md#attention-tensor-layouts) records
200 adjacent pairs and fresh standard trials. The paired changes are +0.5%
for tiny, -1.1% for small, and +0.3% for medium. Tiny has the same attention
operations in both paths. Small uses an extra 24 MiB; medium uses an extra
1.875 MiB. These are the measured costs of the layout fix.

Three small-model story runs preserve their complete learning traces. The
largest final loss change is +0.0278 nats at seed 1337. The report gives both
final and best validation losses for all three seeds.

## Two queued training steps — 2026-09-30

Source `29b81ad` passed 205 tests on Apple M4 Max / MLX 0.32.3. The suite
covers queued/synchronous update parity on CPU and GPU, gradient accumulation,
changing learning rates, report/evaluation boundaries, optimizer and batch RNG
restore, SIGINT/SIGTERM, time limits, and preservation of the last checkpoint
after an error. Ruff checks and formatting pass. The site receipt checks and
six JavaScript layout tests pass with 63 measured configurations.

The standard GPU measurements compare `4566801` with `29b81ad`. The latter's
source hash is `6e77572946aef11d0f67c60330b8fa769bec66fc08367a9b0d9ae8e19aa81ea5`.
Both source trees were clean. The current CPU control is also recorded at
`29b81ad`. Its median is 24,386 bytes/s.

The alternating comparison ran 100 pairs of ten-step blocks with both models
alive and a 4 GiB memory/cache limit. Median queued/reference speed gains are
25.0% for tiny, 6.0% for small, and 1.6% for medium. Single-model peak active
memory rises by 125.4, 547.1, and 63.0 MiB, respectively. `--sync` selects the
lower-memory execution path. All timing sums include completed GPU work.

The 300-step story learning comparison covers tiny seed 1337 and small seeds
1337, 17, and 42. Final validation losses differ by 0.000003 for tiny and at
most 0.0056 for small. The benchmark protocol includes the complete loss and
memory tables and links every raw receipt.

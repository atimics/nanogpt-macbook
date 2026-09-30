# Benchmark protocol

The [benchmark site](https://atimics.github.io/nanogpt-macbook/) reports measured
training performance for this project. Raw receipts live in `results/`. The
learning check has a separate receipt in `learning/demo.json`.

## Repeat the measurements

Use a native Apple Silicon Python and the locked dependencies:

```bash
uv sync --frozen
uv run nanogpt doctor
uv run nanogpt benchmark --preset all --device gpu \
  --steps 100 --warmup 20 --repeats 3 --out benchmarks/results/my-mac-gpu.json
uv run nanogpt benchmark --preset tiny --device cpu \
  --steps 100 --warmup 20 --repeats 3 --out benchmarks/results/my-mac-cpu.json
```

Each trial starts with a fresh model and optimizer, using seed 1337. Batches
contain uniform random byte token IDs. Each step creates a batch on the host,
runs the GPT, computes cross entropy and gradients, clips gradients, updates
weights with AdamW, and checks the loss and gradient norm. Weights and optimizer
state use float32. The first source commit uses eager MLX execution. Later
commits compile the full training step. Commit `29b81ad` queues up to two GPU
steps, then waits for all device work before each trial ends. Earlier GPU
commits and CPU runs wait after each step. The receipt records the execution
mode. Preset context and batch sizes apply.

Commit `17cb28a` adds fused Metal kernels for the attention softmax and its
gradient. Its cache limit is 2 GiB, matching the allocator setting, so temporary
buffers can be reused. Earlier commits set the cache limit to 0.5 GiB. New
receipts record the cache limit and attention path. Weights, optimizer math,
model sizes, batch sizes, and the timed workload keep the same settings.

Commit `f635c16` adds an explicit GELU derivative for float32 Metal training.
Its forward activation matches MLX's approximate GELU. The derivative uses
the stable cosh formula and compiles into fewer elementwise operations.
CPU training uses MLX's native derivative. Receipts record the activation path.

Each trial warms up for 20 steps, then times 100 steps. We run three trials in
sequence. Throughput is `batch * context * timed_steps / sum(step_seconds)`.
The site shows the median trial throughput and the minimum-to-maximum trial
range. Step time is the median of each trial's mean step time. The queued
`training-loop-v2` protocol records the time between checked loss results in
`step_seconds`, including the final device wait in the last interval. These
intervals measure completion spacing. Their sum includes host batch creation,
all GPU updates, every loss/norm check, and the final queue drain. The earlier
`training-step-v1` intervals each include a complete synchronized step.

Evaluation, checkpoint writes, model setup, and warmup sit outside the timer.
The separate demo run measures actual training on text, with held-out loss.
Use that result to check learning and the synthetic runs to compare speed.

Peak memory is MLX's peak active allocation across the measured steps, including
the model and optimizer. The counter resets after warmup. Each trial clears the
allocator cache before model setup. The largest trial peak is reported in MiB.
System memory, Python memory, file mappings, and other apps add to total memory
use. The GPU allocator is configured with a 2 GiB limit; transient allocations
can exceed that value, as recorded in the medium-preset receipt.

## Reading the results

The report tracks source commits on one Apple M4 Max with 36 GiB memory. GPU
presets and the tiny CPU comparison ran sequentially on the same Mac. Each
timeline point is a source commit and one measured preset/device pair. The
same model settings, software stack, seed, and trial counts apply across the
timeline. Power state, temperature, and other apps can change results. Use
the trial range when judging small differences. The execution mode changes
with the compiled training step and the two-step queue. The demo learning
receipt checks training quality on a separate text run.

The `0eb0273` measurements are a fresh baseline taken before the `17cb28a`
speed change. They ran in the same session with the same protocol. The Mac
delivered higher throughput in this session than in the earlier recordings.
Use that pair to assess the attention change; the older points show the
history of measurements under their recorded conditions.

The `5802058` and `f635c16` measurements form the next comparison, for the GELU
derivative. Both ran in one session using the same protocol. Small's GPU trial
ranges are separated. Tiny and medium have overlapping ranges, so their median
differences need care. Peak active memory fell for every GPU preset.

Each JSON receipt includes the source commit, source-file hash, software
versions, model configuration, method, and every measured step. The site builder
recalculates all displayed summaries and checks the stated trial settings.
Published receipts use committed source. The site compares a single machine
and software stack. To contribute another machine, open a PR with the receipt
and a separate clearly labeled report; the builder checks this boundary.

## Build the page

```bash
python scripts/build_site.py
python -m http.server --directory _site 8765
```

The site uses local CSS, JavaScript, and an SVG plot. GitHub Actions builds the
page on pull requests and deploys `main` to GitHub Pages. The site and source
use the MIT-0 license.

## LayerNorm comparison

The LayerNorm update combines four rows of weight and bias gradients in each
Metal threadgroup. Float32 training uses smaller partial-sum buffers. The
forward calculation uses native MLX. CPU training also uses native MLX gradients.
The kernel applies to widths up to 512, including all three presets.

The normal time series includes a fresh baseline at `9060ccc`, GPU results at
`b3380d8`, and a CPU control at `2c34805`. The last two commits share the same
training-source hash. These runs showed large timing swings, including a 7.9%
fall in CPU throughput with the same native CPU math. The page keeps their raw
measured speeds and trial ranges.

For a closer comparison, run both paths in adjacent complete training steps:

```bash
uv run python scripts/compare_normalization.py \
  --out benchmarks/diagnostics/my-layernorm.json
```

This diagnostic keeps two models alive and uses a 4 GiB allocator and cache
limit. Each path gets 20 warmup steps. It then measures 200 pairs, with one
complete step per path in each pair. The first path alternates on each pair.
Model settings, batches, seeds, float32 math, and step synchronization match.
The reported change is the median grouped/native throughput ratio minus one.
This protocol has a separate receipt from the time-series protocol.

| Preset | Median change | Pairs with a gain | Peak memory before / after |
| --- | ---: | ---: | ---: |
| tiny / Metal | +4.2% | 119 / 200 | 149.4 / 146.9 MiB |
| small / Metal | -0.4% | 95 / 200 | 1033.1 / 1011.0 MiB |
| medium / Metal | +3.3% | 181 / 200 | 2033.1 / 1907.5 MiB |

Memory comes from the standard single-model benchmark. The adjacent-step
comparison gives the clearest speed gain for medium. Tiny has more variation;
small is roughly level. Ratios from total elapsed time were +1.4%, -1.2%, and
+3.5%, respectively. Both calculations are available in the raw timings.

The [adjacent-step receipt](diagnostics/layernorm-2c34805-adjacent.json) records
every measured step. The earlier [25-step block comparison](diagnostics/layernorm-efebdd0-blocks.json)
also remains available. Its median changes were +7.5%, -2.4%, and +2.8%; shorter
pairs reduce the time between each comparison. The story learning check reached
validation loss 2.1712 after 300 steps. Its [receipt](learning/demo-2c34805.json)
records the source hash and metrics.

## Gradient clipping comparison

Commit `17ba7ea` groups up to fourteen float32 gradient arrays in each Metal
reduction. It reduces the kernel launches needed to find the global gradient
norm. The clipping scale and epsilon match MLX. CPU clipping uses native MLX.

The standard time series includes a fresh baseline at `fa60b54` and the update
at `17ba7ea`. Both use the usual 20 warmup steps and three 100-step trials:

| Preset | Baseline bytes/s | Updated bytes/s | Median change | Peak memory before / after |
| --- | ---: | ---: | ---: | ---: |
| tiny / Metal | 368,281 | 372,007 | +1.0% | 146.9 / 144.2 MiB |
| small / Metal | 140,803 | 143,437 | +1.9% | 1011.0 / 1033.0 MiB |
| medium / Metal | 50,981 | 51,129 | +0.3% | 1907.5 / 2034.9 MiB |

Tiny and medium have overlapping trial ranges. Small's ranges are separated.
Grouping gradients keeps more arrays alive at once: small uses 22.0 MiB more
peak memory and medium uses 127.5 MiB more. Tiny uses 2.7 MiB less.

The adjacent-step comparison uses the same protocol as the LayerNorm comparison
above, with two live models and a 4 GiB memory/cache limit:

```bash
uv run python scripts/compare_clipping.py \
  --out benchmarks/diagnostics/my-clipping.json
```

| Preset | Median change | Total-time change | Pairs with a gain |
| --- | ---: | ---: | ---: |
| tiny / Metal | +3.5% | +3.7% | 188 / 200 |
| small / Metal | +1.4% | +1.4% | 186 / 200 |
| medium / Metal | +2.6% | +2.5% | 193 / 200 |

The [adjacent-step receipt](diagnostics/clipping-17ba7ea-adjacent.json) includes
every measured step. The [300-step learning receipt](learning/demo-17ba7ea.json)
records validation loss 2.1712059, matching the prior 2.1712157 to four decimal
places. The GPU measurements use committed source. The latest CPU timeline
point retains its own measured source commit.

## Saved LayerNorm statistics

Commit `231140f` saves the row statistics from LayerNorm's forward pass and
reuses them for gradients. It combines sixteen rows per backward group for
widths up to 256 and eight rows for wider supported inputs. This reduces
partial gradient buffers. Each row is shifted by its first value before the
mean is calculated, which preserves small differences in nearly constant rows.
CPU normalization and the checkpoint layout keep their existing behavior.

The standard time series includes the baseline at `63a55cb` and this update:

| Preset | Baseline bytes/s | Updated bytes/s | Peak memory before / after |
| --- | ---: | ---: | ---: |
| tiny / Metal | 354,993 | 375,699 | 144.2 / 142.6 MiB |
| small / Metal | 74,453 | 139,522 | 1033.0 / 1022.9 MiB |
| medium / Metal | 27,717 | 50,361 | 2034.9 / 2022.7 MiB |

Machine timing varied sharply between these sequential runs. The adjacent-step
comparison gives a closer estimate of this code change. It loads LayerNorm
from the trusted local baseline commit and keeps all other training code fixed:

```bash
uv run python scripts/compare_normalization.py --baseline-ref 63a55cb \
  --out benchmarks/diagnostics/my-saved-normalization.json
```

The receipt includes the full baseline commit and the hash of the loaded file.
Both paths get 20 warmup steps, followed by 200 pairs of complete training
steps. Their order alternates each pair. The comparison keeps two models alive
and uses a 4 GiB memory/cache limit; the standard protocol uses one model and
2 GiB. Settings, batches, seeds, optimizer, and float32 precision match.

| Preset | Median change | Total-time change | Pairs with a gain |
| --- | ---: | ---: | ---: |
| tiny / Metal | +4.2% | +3.7% | 126 / 200 |
| small / Metal | +0.6% | +0.6% | 124 / 200 |
| medium / Metal | +1.0% | +1.2% | 168 / 200 |

The [paired receipt](diagnostics/normalization-231140f-adjacent.json) records
every step. These gains are modest, especially for small. Standard peak
memory fell by 1.6 MiB for tiny, 10.2 MiB for small, and 12.2 MiB for medium.
The [learning check](learning/demo-231140f.json) reached validation loss
2.1712095 after 300 steps, matching the prior 2.1712059 to four decimal places.

## Attention score gradients

Commit `ddb8a6f` combines the attention score-gradient matrix product and
softmax derivative in one Metal kernel. It calculates the row reduction from
the saved attention output. Masked matrix products skip complete blocks above
the causal boundary. The path covers float32 tensors with matching batch/head
shapes, contexts from 256 to 512 in multiples of 32, and head widths from 32 to
64 in multiples of eight. Small and medium use this path. Tiny uses the existing
fused-softmax path.

The standard protocol compares baseline `0898920` with `ddb8a6f`:

| Preset | Baseline bytes/s | Updated bytes/s | Peak memory before / after |
| --- | ---: | ---: | ---: |
| tiny / Metal | 395,332 | 388,234 | 142.6 / 142.6 MiB |
| small / Metal | 144,809 | 157,647 | 1022.9 / 975.3 MiB |
| medium / Metal | 52,966 | 57,027 | 2022.7 / 1882.4 MiB |

The alternating comparison holds the rest of the current training code fixed
and loads attention from the baseline commit. Both models receive the same
seeded batches. Each path gets 20 warmup steps and 200 measured steps, with
the order reversed each pair. This uses a 4 GiB memory/cache limit for two
live models; the standard single-model runs use 2 GiB.

```bash
uv run python scripts/compare_attention.py --baseline-ref 0898920 \
  --out benchmarks/diagnostics/my-attention.json
```

| Preset | Median change | Total-time change | Pairs with a gain |
| --- | ---: | ---: | ---: |
| tiny / Metal | -0.7% | -0.5% | 85 / 200 |
| small / Metal | +9.1% | +9.1% | 200 / 200 |
| medium / Metal | +6.0% | +5.5% | 191 / 200 |

The [paired receipt](diagnostics/attention-ddb8a6f-adjacent.json) records all
steps, the baseline file hash, and both summaries. Tiny retains the same GPU
operations; its measured difference shows timing variation between paths.
Peak active memory fell by 47.6 MiB for small and 140.4 MiB for medium.

Float32 operations are grouped differently in the new gradient. Compiled
gradients match a float64 reference within `atol=5e-6, rtol=5e-5`; full-model
gradients and three accumulated AdamW updates match native MLX within
`atol=2e-6, rtol=2e-5`. Causal masking and checkpoint resume also pass.

Small-model training on the bundled story was checked with three seeds. Each
run uses the default preset, 300 steps, and ten validation batches every 100
steps. The six receipts preserve each learning trace:

| Seed | Baseline final validation loss | Updated final validation loss | Difference |
| --- | ---: | ---: | ---: |
| 1337 | [2.218576](learning/small-0898920-seed1337.json) | [2.257929](learning/small-ddb8a6f-seed1337.json) | +0.039353 |
| 17 | [2.299735](learning/small-0898920-seed17.json) | [2.299945](learning/small-ddb8a6f-seed17.json) | +0.000210 |
| 42 | [2.243085](learning/small-0898920-seed42.json) | [2.246769](learning/small-ddb8a6f-seed42.json) | +0.003683 |

Both paths learn from initial loss around 5.6 nats. Their results agree within
three millionths at step 100, then diverge with further updates. The updated
final losses are slightly higher in all three runs. The CLI samples from the
saved best checkpoint by default. Those checkpoints have these losses:

| Seed | Baseline best step / loss | Updated best step / loss |
| --- | ---: | ---: |
| 1337 | 300 / 2.218576 | 200 / 2.207986 |
| 17 | 200 / 2.197994 | 200 / 2.197998 |
| 42 | 200 / 2.204283 | 200 / 2.204979 |

These short runs on a
4,544-byte story describe this training check; broader text quality needs a
larger evaluation. The refreshed [tiny learning receipt](learning/demo-ddb8a6f.json)
reaches 2.1712084 after 300 steps.

## Attention tensor layouts

Source `974dde6` explicitly packs Q, K, V, and the incoming gradient before
masked products. In MLX 0.32.3, the Metal masked-matrix implementation can copy
an operand and then use batch strides from the original input. Width slices
and reversed views expose incorrect values. Explicit packing gives each
product a consistent layout. The [pinned MLX implementation](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/matmul.cpp#L1805)
shows the copy and batch-stride handling.

The new regression fails on `95eb883`. All 185 local tests pass after the fix,
including 16 new cases covering projection views, width slices, reversed
views, and broadcast inputs in eager and compiled calls. Each case compares
outputs and all three gradients with native attention on contiguous operands.
Existing checks cover float64 gradients, full model updates, accumulation,
and checkpoint resume.

The [200-pair receipt](diagnostics/attention-974dde6-adjacent.json) compares
complete training steps with the attention at `95eb883`. It uses the same
adjacent-step protocol as the score-gradient comparison:

| Preset | Median throughput change | Total-time throughput change | Faster pairs |
| --- | ---: | ---: | ---: |
| tiny | +0.5% | +0.6% | 110 / 200 |
| small | -1.1% | -1.0% | 30 / 200 |
| medium | +0.3% | +0.3% | 136 / 200 |

Tiny uses the same attention operations in both paths. Its difference shows
short-run timing variation. Small pays about 1% for explicit packing. This
change fixes tensor-layout correctness; further speed work uses this path as
the baseline.

Repeat the paired comparison with:

```bash
uv run python scripts/compare_attention.py --baseline-ref 95eb883 \
  --preset all --pairs 200 --steps 1 --warmup 20 \
  --out benchmarks/diagnostics/attention-layouts-local.json
```

The [baseline](results/m4-max-95eb883-gpu.json) and
[updated](results/m4-max-974dde6-gpu.json) standard receipts use one model,
a 2 GiB memory/cache limit, and three trials of 100 measured steps:

| Preset | Baseline bytes/s | Updated bytes/s | Baseline peak MiB | Updated peak MiB |
| --- | ---: | ---: | ---: | ---: |
| tiny | 389,597 | 390,409 | 142.6 | 142.6 |
| small | 156,706 | 154,739 | 975.3 | 999.3 |
| medium | 55,744 | 56,227 | 1882.4 | 1884.2 |

Explicit packing adds 24 MiB for small and 1.875 MiB for medium in these runs.
The medium trial ranges overlap. Both receipts remain in the commit history.

Three 300-step small-model story runs use the same seeds and settings as the
prior learning checks. The baseline receipts at `ddb8a6f` share the training
source hash of `95eb883`. Both final and best measured validation losses follow:

| Seed | Baseline final | Updated final | Difference | Baseline best | Updated best |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1337 | 2.257929 | [2.285701](learning/small-974dde6-seed1337.json) | +0.027773 | 2.207986 | 2.228266 |
| 17 | 2.299945 | [2.299168](learning/small-974dde6-seed17.json) | -0.000777 | 2.197998 | 2.197990 |
| 42 | 2.246769 | [2.244856](learning/small-974dde6-seed42.json) | -0.001913 | 2.204979 | 2.204514 |

The largest final difference is +0.0278 nats for seed 1337. All best values
occur at step 200. Packing changes the physical layout used by matrix
operations, and small float32 differences grow over later updates. These
short checks use a 455-byte validation split; broader text quality needs
a larger evaluation. The receipts retain every reported value.

## Two queued training steps

Commit `29b81ad` queues up to two compiled GPU updates. Python can prepare the
next batch while Metal works. Each loss and gradient norm is checked, and the
queue finishes before reports, evaluation, checkpoints, and shutdown. Training
and the standard GPU benchmark use the queue by default. `--sync` selects a
wait after each step and reduces peak active memory.

The fresh standard comparison uses `4566801` and `29b81ad`, with one model,
a 2 GiB memory/cache limit, 20 warmup steps, and three 100-step trials:

| Preset | Baseline bytes/s | Queued bytes/s | Peak memory before / after |
| --- | ---: | ---: | ---: |
| tiny / Metal | 393,106 | 482,575 | 142.6 / 268.0 MiB |
| small / Metal | 156,853 | 166,338 | 999.3 / 1546.4 MiB |
| medium / Metal | 58,454 | 60,012 | 1884.2 / 1947.2 MiB |

The current tiny CPU control records 24,386 bytes/s with per-step waits. CPU
math uses the same path; its speed also reflects the conditions of this run.

Tiny and small have separated trial ranges. Medium ranges overlap, with more
variation in its queued run. The queue keeps more arrays active at once.
Peak active memory rises by 125.4 MiB for tiny, 547.1 MiB for small, and
63.0 MiB for medium. Raw results include every measured interval.

For a closer speed comparison, alternate the two paths in ten-step blocks:

```bash
uv run python scripts/compare_pipeline.py --baseline-ref 4566801 \
  --out benchmarks/diagnostics/my-training-queue.json
```

This diagnostic loads the reference compiled step from the trusted local
commit. It uses the current model math for both paths, matching the reference
commit's model math in this comparison. The receipt stores the reference file
hash and source commit. Two models stay alive with a 4 GiB memory/cache limit.
Each path gets 20 warmup steps, followed by 100 pairs of ten-step blocks.
The order alternates each pair. Timings include batch creation, training,
loss/norm checks, and all device work through each block's final wait.

| Preset | Median speed gain | Total-time gain | Pairs with a gain |
| --- | ---: | ---: | ---: |
| tiny / Metal | +25.0% | +24.9% | 100 / 100 |
| small / Metal | +6.0% | +6.2% | 100 / 100 |
| medium / Metal | +1.6% | +1.8% | 87 / 100 |

The [paired receipt](diagnostics/pipeline-29b81ad-blocks.json) includes each
completion interval and throughput ratio. The default training loop reports
every ten steps, which also drains the queue at ten-step boundaries.

Learning checks compare synchronous and queued training on the bundled story
for 300 steps with the same float32 settings, batches, and learning rates:

| Preset / seed | Synchronous final loss | Queued final loss | Difference |
| --- | ---: | ---: | ---: |
| tiny / 1337 | 2.1712080 | 2.1712108 | +0.0000029 |
| small / 1337 | 2.2888015 | 2.2901658 | +0.0013643 |
| small / 17 | 2.2982055 | 2.2966031 | -0.0016025 |
| small / 42 | 2.2469623 | 2.2414097 | -0.0055527 |

These are final held-out losses in nats. The largest absolute final difference
for small is 0.0056; its best validation step is 200 in both modes for all three
seeds. Floating-point results can vary as GPU execution changes. The paired
receipts for [tiny](learning/pipeline-29b81ad-tiny-sync.json),
[queued tiny](learning/pipeline-29b81ad-tiny-queued.json),
[small 1337](learning/pipeline-29b81ad-small-sync.json),
[queued small 1337](learning/pipeline-29b81ad-small-queued.json),
[small 17](learning/pipeline-29b81ad-small-sync-seed17.json),
[queued small 17](learning/pipeline-29b81ad-small-queued-seed17.json),
[small 42](learning/pipeline-29b81ad-small-sync-seed42.json), and
[queued small 42](learning/pipeline-29b81ad-small-queued-seed42.json)
include all validation checkpoints. Focused tests also compare weights,
optimizer state, batch RNG, accumulation, stop handling, and resume behavior.

## Fused attention forward pass

Commit `ac55ea0` combines the score matrix product and causal softmax for the
small and medium Metal presets. Each threadgroup calculates eight query rows,
normalizes the score tile in shared memory, and writes attention probabilities.
This saves an intermediate score tensor and a kernel launch. The existing
backward pass uses the saved probabilities. Float32 precision and the two-step
training queue apply to both sides of the comparison.

The fresh standard comparison uses `dccdbce` and `ac55ea0`, with one model,
a 2 GiB memory/cache limit, 20 warmup steps, and three 100-step trials:

| Preset | Baseline bytes/s | Updated bytes/s | Peak memory before / after |
| --- | ---: | ---: | ---: |
| tiny / Metal | 485,082 | 480,910 | 268.0 / 268.0 MiB |
| small / Metal | 166,335 | 173,784 | 1546.4 / 1454.8 MiB |
| medium / Metal | 60,203 | 63,116 | 1947.2 / 1835.8 MiB |

Small's trial ranges are separated. Medium's ranges overlap. Peak active
memory falls by 91.6 MiB for small and 111.4 MiB for medium. Tiny uses its
existing attention path and serves as a timing control.

The paired comparison runs both attention implementations with queued updates:

```bash
uv run python scripts/compare_attention.py --baseline-ref dccdbce \
  --queued --pairs 100 --steps 10 --warmup 20 \
  --out benchmarks/diagnostics/my-attention-forward.json
```

Both models stay alive with a 4 GiB memory/cache limit. Their order alternates
across 100 pairs of ten-step blocks. Each path gets 20 warmup steps. All loss
and gradient-norm checks and the final GPU wait sit inside the timed blocks.
The reference attention file, its hash, source commits, and all completion
intervals are recorded in the [paired receipt](diagnostics/attention-forward-ac55ea0-blocks.json).

| Preset | Median speed change | Total-time change | Pairs with a gain |
| --- | ---: | ---: | ---: |
| tiny / Metal | 0.0% | 0.0% | 50 / 100 |
| small / Metal | +4.5% | +4.3% | 98 / 100 |
| medium / Metal | +3.0% | +3.1% | 95 / 100 |

The suite passes 217 tests on Apple Silicon. Twelve new cases compare the
saved probabilities with a float64 reference across uniform, random, and
larger scores. They check row sums and exact zeros above the causal boundary.
Existing tests cover gradients against float64 and MLX, partial block sizes,
sliced/reversed/broadcast tensors, compiled model updates, accumulation,
queued training, stopping, and checkpoint resume.

The learning check compares 300 queued steps on the bundled story for three
small-model seeds. Final held-out loss is measured in nats:

| Seed | Baseline final | Updated final | Difference |
| --- | ---: | ---: | ---: |
| 1337 | 2.2982645 | 2.2906660 | -0.0075985 |
| 17 | 2.2989750 | 2.3002633 | +0.0012883 |
| 42 | 2.2451372 | 2.2442262 | -0.0009109 |

Both paths reach their best validation checkpoint at step 200 for each seed.
The largest absolute final difference is 0.0076 nats. Best-checkpoint losses
differ by at most 0.0022 nats. These short runs check
learning behavior on the bundled story. Broader text quality needs broader data.
The complete traces are in the baseline and updated receipts:

- Seed 1337: [baseline](learning/attention-forward-dccdbce-small-seed1337.json), [updated](learning/attention-forward-ac55ea0-small-seed1337.json).
- Seed 17: [baseline](learning/attention-forward-dccdbce-small-seed17.json), [updated](learning/attention-forward-ac55ea0-small-seed17.json).
- Seed 42: [baseline](learning/attention-forward-dccdbce-small-seed42.json), [updated](learning/attention-forward-ac55ea0-small-seed42.json).

## Fused residual projections

Commit `12beb3c` combines a GPU matrix projection with its residual addition
using native MLX `addmm`. Float32 training uses this operation for projections
with up to 1024 input features. The medium model uses it for attention and uses
separate operations for its wider MLP projection. Float32 GPU evaluation uses
the fused operation for both projections. CPU training uses separate operations.
The parameter names and checkpoint layout stay compatible.

Measurements use an M4 Max, MLX 0.32.3, float32, and two queued training steps.
The baseline is `09a5066`. Fresh standard runs use one model, a 2 GiB memory/cache
budget, 20 warmup steps, and three 100-step trials. Both complete receipts are
available: [baseline](results/m4-max-09a5066-gpu.json) and
[updated](results/m4-max-12beb3c-gpu.json).

| Preset | Baseline bytes/s | Updated bytes/s | Change | Peak memory before / after |
| --- | ---: | ---: | ---: | ---: |
| tiny / Metal | 483,276 | 490,870 | +1.6% | 268.0 / 279.5 MiB |
| small / Metal | 147,012 | 144,480 | -1.7% | 1454.8 / 1519.0 MiB |
| medium / Metal | 52,517 | 52,663 | +0.3% | 1835.8 / 1844.8 MiB |

All three standard trial ranges overlap. Tiny and medium have higher medians;
small has a lower median in these separate runs. The paired checks below measure
adjacent blocks to compare both paths under nearby machine conditions. Peak
active memory rises by 11.5 MiB for tiny, 64.1 MiB for small, and 9.0 MiB for medium.

```bash
uv run python scripts/compare_model.py --baseline-ref 09a5066 \
  --queued --pairs 100 --steps 10 --warmup 20 --memory-gb 2 \
  --out benchmarks/diagnostics/my-residual-2gib.json
```

Each paired check keeps two complete models alive. Both get 20 warmup steps,
then 100 pairs of ten-step blocks. Their order alternates each pair. Timings
include batch creation, forward and backward work, clipping, AdamW, loss/norm
checks, and the final GPU wait. The same comparison also runs with 4 GiB.

| Preset | Budget | Median gain | Total-time gain | Pairs with a gain |
| --- | ---: | ---: | ---: | ---: |
| tiny / Metal | 2 GiB | +5.6% | +5.5% | 100 / 100 |
| small / Metal | 2 GiB | +2.4% | +2.2% | 89 / 100 |
| medium / Metal | 2 GiB | +1.1% | +1.2% | 96 / 100 |
| tiny / Metal | 4 GiB | +5.4% | +5.5% | 100 / 100 |
| small / Metal | 4 GiB | +1.5% | +1.6% | 78 / 100 |
| medium / Metal | 4 GiB | +1.0% | +1.3% | 92 / 100 |

The [2 GiB receipt](diagnostics/residual-d67322d-2gib-blocks.json) and
[4 GiB receipt](diagnostics/residual-12beb3c-blocks.json) include all intervals,
source commits, and the reference model file hash. The 2 GiB receipt uses
`d67322d`, which adds the budget option to the comparison script. Both updated
runs have the same Python package source hash.

All 241 tests pass on Apple Silicon. The 24 new CPU/GPU cases compare logits,
loss, every parameter gradient, and queued AdamW updates with separate residual
additions. They cover partial and strided inputs, all preset widths, evaluation,
accumulation, and changing learning rates. Existing checks cover causal attention,
training stops, checkpoint loading, and resume.

Five matched learning comparisons run 300 steps on the bundled story. Tiny and
small use a 10% validation split (455 bytes). Medium uses a fixed 20% split
(909 bytes), which fits its 512-byte context. Each pair uses matching data, seed,
schedule, and float32 settings. Loss is held-out cross entropy in nats.

| Preset / seed | Baseline final | Updated final | Final difference | Best-loss difference |
| --- | ---: | ---: | ---: | ---: |
| tiny / 1337 | 2.1712088 | 2.1712077 | -0.0000011 | -0.0000011 |
| small / 1337 | 2.2341930 | 2.2763444 | +0.0421514 | -0.0058472 |
| small / 17 | 2.3008793 | 2.2963816 | -0.0044977 | -0.0000614 |
| small / 42 | 2.2464574 | 2.2431281 | -0.0033293 | -0.0005926 |
| medium / 1337 | 2.9039253 | 2.9038767 | -0.0000486 | +0.0000000 |

The largest absolute final-loss difference is 0.0422 nats; the largest
best-checkpoint difference is 0.00585 nats. Both paths choose the same best steps:
300 for tiny, 200 for all small seeds, and 100 for medium. These short-story
runs check the training path. Broader text quality needs broader data. Each
receipt includes the preparation command, data hashes, and all validation points.

- tiny / 1337: [baseline](learning/residual-09a5066-tiny-seed1337.json), [updated](learning/residual-12beb3c-tiny-seed1337.json).
- small / 1337: [baseline](learning/residual-09a5066-small-seed1337.json), [updated](learning/residual-12beb3c-small-seed1337.json).
- small / 17: [baseline](learning/residual-09a5066-small-seed17.json), [updated](learning/residual-12beb3c-small-seed17.json).
- small / 42: [baseline](learning/residual-09a5066-small-seed42.json), [updated](learning/residual-12beb3c-small-seed42.json).
- medium / 1337: [baseline](learning/residual-09a5066-medium-seed1337.json), [updated](learning/residual-12beb3c-medium-seed1337.json).

## Complete attention forward pass

Source `1203504` combines scores, causal softmax, and the value product in one
Metal kernel. It covers context 128 and aligned contexts 256–512 with head
widths 32–64. Each tile handles eight queries. Shorter contexts use 128 GPU
threads; longer contexts use 256 and store output in the order consumed by
the projection. Saved probabilities support the existing gradient kernel and
masked products. Model settings, float32 math, and the two-step queue match
the baseline at `316f010`.

The comparison uses a 2 GiB allocator and cache setting. The first diagnostic
alternates 100 pairs of ten-step blocks while both models stay alive. The
second alternates 20 pairs of fresh models. Each fresh model runs 20 warmup
steps and 100 timed steps; one model stays alive at a time. Every timed block
includes its final GPU wait.

| Preset | Ten-step median gain | Faster blocks / 100 | Fresh-trial median gain | Fresh total-time gain | Faster fresh trials / 20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| tiny | +2.9% | 100 | +3.1% | +3.0% | 20 |
| small | +1.0% | 81 | +0.5% | +0.7% | 15 |
| medium | +1.1% | 97 | +0.2% | +1.0% | 11 |

Raw diagnostics: [ten-step blocks](diagnostics/attention-values-1203504-blocks.json)
and [fresh-model trials](diagnostics/attention-values-1203504-fresh.json).
The fresh receipt also records peak active memory for each trial.
Tiny has the clearest gain. Small has a modest gain. Medium's fresh results
are close to even, with 11 of 20 pairs faster.

```bash
uv run python scripts/compare_attention.py --baseline-ref 316f010 \
  --queued --memory-gb 2 --pairs 100 --steps 10 --warmup 20 \
  --out benchmarks/diagnostics/attention-blocks.json
uv run python scripts/compare_attention.py --baseline-ref 316f010 \
  --queued --fresh --memory-gb 2 --pairs 20 --steps 100 --warmup 20 \
  --out benchmarks/diagnostics/attention-fresh.json
```

The time series retains the separate three-trial measurements. These ran
sequentially across presets and source versions. Their medians and ranges
record conditions during those runs. The alternating trials provide a closer
comparison of the implementations.

| Preset | Baseline median bytes/s | Final median bytes/s | Recorded change | Peak active MiB before → after |
| --- | ---: | ---: | ---: | ---: |
| tiny | 508,279 | 474,823 | -6.6% | 279.5 → 275.6 |
| small | 170,085 | 124,744 | -26.7% | 1,519.0 → 1,525.0 |
| medium | 59,244 | 45,758 | -22.8% | 1,844.8 → 1,886.4 |

- tiny: baseline range 508,084–512,573 bytes/s; final range 461,522–480,768.
- small: baseline range 167,859–172,838 bytes/s; final range 121,792–125,288.
- medium: baseline range 58,621–60,304 bytes/s; final range 43,154–46,082.

Raw standard receipts: [baseline](results/m4-max-316f010-gpu.json),
[first fused version](results/m4-max-1d71ed2-gpu.json), and
[final version](results/m4-max-1203504-gpu.json). All three remain on the time
series. The first fused version kept its longer outputs in attention order.
Its medium fresh-model median gained 0.2%, with 11 of 20 faster pairs and
a 0.9% fall in aggregate throughput. It used 65.6 MiB more peak memory.
The final output layout saves 24 MiB relative to that version. Its
[initial paired](diagnostics/attention-values-1d71ed2-blocks.json) and
[initial fresh](diagnostics/attention-values-c39387d-fresh.json) receipts
remain available.

Five matched 300-step learning checks use the bundled story. Tiny and small
use the existing 90/10 split; medium uses 80/20 so the validation split fits
its 512-byte context. Each comparison keeps the preset, seed, optimizer,
queue, evaluation schedule, and data fixed. These short checks describe
training on this story.

| Preset, seed | Final validation loss before → after | Best validation loss before → after | Best step |
| --- | ---: | ---: | ---: |
| tiny, 1337 | [2.171206](learning/attention-values-316f010-tiny-seed1337.json) → [2.171205](learning/attention-values-1203504-tiny-seed1337.json) | 2.171206 → 2.171205 | 300 |
| small, 1337 | [2.278083](learning/attention-values-316f010-small-seed1337.json) → [2.251307](learning/attention-values-1203504-small-seed1337.json) | 2.219387 → 2.219491 | 200 |
| small, 17 | [2.301266](learning/attention-values-316f010-small-seed17.json) → [2.295916](learning/attention-values-1203504-small-seed17.json) | 2.198031 → 2.197936 | 200 |
| small, 42 | [2.240472](learning/attention-values-316f010-small-seed42.json) → [2.239471](learning/attention-values-1203504-small-seed42.json) | 2.204096 → 2.204035 | 200 |
| medium, 1337 | [2.904349](learning/attention-values-316f010-medium-seed1337.json) → [2.904220](learning/attention-values-1203504-medium-seed1337.json) | 2.421399 → 2.421400 | 100 |

The largest absolute final-loss difference is 0.0268 nats; the
largest best-checkpoint difference is 0.0002 nats. Each pair selects
the same best checkpoint step. Initial-version learning receipts are also
preserved: [tiny 1337](learning/attention-values-1d71ed2-tiny-seed1337.json), [small 1337](learning/attention-values-1d71ed2-small-seed1337.json), [small 17](learning/attention-values-1d71ed2-small-seed17.json), [small 42](learning/attention-values-1d71ed2-small-seed42.json), [medium 1337](learning/attention-values-1d71ed2-medium-seed1337.json).

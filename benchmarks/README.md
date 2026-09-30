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
weights with AdamW, and waits for device work to finish. Weights and optimizer
state use float32. The first source commit uses eager MLX execution. Later
commits compile the full training step. The receipt records the execution mode.
Preset context and batch sizes apply.

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
range. Step time is the median of each trial's mean step time.

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
with the compiled training step. The demo learning receipt checks training
quality on a separate text run.

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

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
Use these two latest commits to assess this change; the older points show the
history of measurements under their recorded conditions.

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

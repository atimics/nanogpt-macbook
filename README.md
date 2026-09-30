# nanoGPT MacBook edition

Train a small GPT from scratch on an Apple Silicon MacBook. This kit uses
[Apple MLX](https://github.com/ml-explore/mlx) for Metal GPU training and offers
a simple command line for text import, training, resume, and text generation.

[Benchmarks](https://atimics.github.io/nanogpt-macbook/) ·
[Raw measurements](benchmarks/results/) · [MIT-0 license](LICENSE)

Inspired by [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT).
The model and training code here are written for MLX. Models learn to continue
text from your own corpus. The bundled story provides a quick first experiment.

## Start here

Use an M-series Mac, macOS 14 or later, and native arm64 Python 3.11 or later.
[Install uv](https://docs.astral.sh/uv/getting-started/installation/) to manage
Python and the project environment.

```bash
git clone https://github.com/atimics/nanogpt-macbook.git
cd nanogpt-macbook
uv sync --frozen
uv run nanogpt doctor
uv run nanogpt prepare --demo --out data/demo
uv run nanogpt train --data data/demo --run runs/first --steps 300
uv run nanogpt sample --run runs/first --prompt "Mira " --tokens 300
```

The first setup downloads Python packages. The commands then run locally.
The source and benchmark site are public under the MIT-0 license. The benchmark
page plots training speed across measured source commits, with raw timing data
for each point. The training loop compiles gradients, clipping, and AdamW into
one MLX step for higher throughput. Metal training queues up to two steps so
the CPU can prepare work while the GPU runs. Every step's loss and gradient
norm are checked. The queue finishes before reports, evaluation, and checkpoints.
The queue uses more peak GPU memory. Use `--sync` with `train`, `resume`, or
`benchmark` to wait after each step and reduce peak memory.

The demo contains a short original story. A short run should reduce the loss
and start to learn letters and word patterns. Use a larger, varied corpus and
longer runs to improve the text. Model quality depends on data, size, and training.

## Train on your text

Save text as UTF-8, then prepare it once:

```bash
uv run nanogpt prepare --input book.txt --out data/book
uv run nanogpt train --data data/book --run runs/book --preset small --steps 2000
uv run nanogpt evaluate --run runs/book
uv run nanogpt sample --run runs/book --prompt "Once upon a time" --temperature 0.8
```

Preparation puts the first 90% of the bytes in `train.bin` and the final 10% in
`val.bin`. Each split must contain more bytes than the model's context length.
Use `--validation-fraction 0.05` to change the split. Preserve document order
when held-out documents matter for your experiment.

Each token is one UTF-8 byte, with a fixed vocabulary of 256. All languages and
symbols can be encoded. A context of 128 tokens covers 128 bytes. Early samples
can contain replacement characters when the model generates partial UTF-8.

Files are memory mapped for training. A manifest records split sizes and SHA-256
hashes. Training checks the files before it starts. Keep prepared data unchanged
while a run is active.

## Choose a size

```bash
uv run nanogpt presets
```

| Preset | Parameters | Layers | Width | Context | Batch |
| --- | ---: | ---: | ---: | ---: | ---: |
| tiny | 837,888 | 4 | 128 | 128 | 8 |
| small | 4,856,320 | 6 | 256 | 256 | 8 |
| medium | 14,463,744 | 8 | 384 | 512 | 4 |

Start with `tiny`. Watch the reported peak MLX memory before increasing the size.
The default GPU memory limit is 2 GiB. MLX's cache limit uses the same setting
so training can reuse temporary buffers across steps.
Use `--memory-gb 4` for a larger budget. Python, data mappings, macOS, and other
apps use additional memory. MLX limits apply to the MLX allocator.

```bash
uv run nanogpt train --data data/book --run runs/compact --steps 1000 \
  --context 64 --batch-size 4 --accumulation 4 --memory-gb 2
```

Gradient accumulation combines several small batches into one optimizer step.
The effective batch is `batch-size * accumulation`. Smaller batches and contexts
reduce memory use. `--device cpu` provides a small CPU test on a supported Mac.
`--device auto` selects Metal when it is available.

## Stop and resume

Press **Ctrl+C** once to finish the queued steps and save. Up to two submitted
steps can finish after the request. A time limit provides
a bounded session:

```bash
uv run nanogpt train --data data/book --run runs/evening --steps 2000 --time-limit 600
uv run nanogpt resume --run runs/evening --steps 2000
```

`--steps` is the total target count. To extend a completed 2,000-step run, use
`resume --steps 3000`. Resume restores model weights, AdamW state, the step,
settings, and random state. `--data` can point to a moved copy of the same data.
The data hashes must match. Repeating a run on the same device and MLX version
supports repeatable results; floating-point results can vary across hardware.

The learning rate warms up for 20 steps, then follows a cosine curve to step
2,000. Later steps use the final rate, one tenth of the starting rate. Extending
a run preserves that schedule. New runs can set `--learning-rate` and `--seed`.

## Saved runs

Each run contains:

- `run.json`: model, data, and training settings.
- `metrics.jsonl`: training loss, speed, memory, and validation results.
- `latest.json`: pointer to the latest complete checkpoint.
- `best.json`: pointer to the checkpoint with the lowest measured validation loss.
- `checkpoints/`: model weights, optimizer state, random state, and metadata.

Checkpoints use safetensors and JSON. A checkpoint is written in a temporary
folder, then published through an atomic pointer replacement. The latest and
best checkpoints are retained to bound disk use. A process lock gives each run
one training writer. Sampling and evaluation load a saved run after training ends.

Validation runs at step zero, every 100 steps, and at the end. Adjust this with
`--eval-every` and `--eval-batches`. Validation uses a fixed sample of the held-out
split so results can be compared. Larger batch counts provide a broader estimate.
The initial untrained checkpoint can remain the best if training raises held-out loss.

`sample` and `evaluate` use the best checkpoint by default. Add
`--checkpoint latest` to inspect the last completed step. `sample --temperature 0`
uses greedy generation. `--top-k 0` samples from the whole vocabulary.

Loss is cross entropy in natural log units. Bits per byte divides that loss by
`log(2)`. Lower values mean better prediction on the measured validation samples.
The CLI reports byte tokens per second and peak MLX memory. Its reported speed
includes work between log points, including validation and saves.

## Design

The GPT uses learned position embeddings, pre-layer normalization, causal
attention, GELU feed-forward layers, and tied input/output embeddings. Weights
and optimizer state use float32. Metal training uses fused kernels for the
causal softmax and its gradient at context lengths up to 512. MLX handles the
matrix operations, inference attention, and CPU training. An explicit GELU
derivative reduces temporary activation storage in compiled Metal training.
AdamW, gradient
clipping, and small defaults make local experiments
easy to inspect. The source is split into model, data, checkpoint, training, and
CLI modules under `src/nanogpt_macbook`.

## Measure your Mac

```bash
uv run nanogpt benchmark --preset all --device gpu \
  --steps 100 --warmup 20 --repeats 3 --out benchmarks/results/my-mac-gpu.json
```

The command records raw step timings, speed, memory, hardware, software versions,
and the source commit. See the [benchmark protocol](benchmarks/README.md) for
the measurement scope and the CPU comparison command.

## Development

```bash
uv sync --frozen
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python -m build
```

The tests cover causal attention, learning, batch accumulation, checkpoint resume,
data integrity, sampling, and CLI workflows. On Apple Silicon, they also compare
the Metal attention outputs and gradients with MLX and check GPU checkpoint resume.
CI uses the MLX CPU package on Linux.
Local Metal validation is recorded in
[`docs/validation.md`](docs/validation.md).

See the [MLX installation guide](https://ml-explore.github.io/mlx/build/html/install.html)
for supported Python and macOS versions.

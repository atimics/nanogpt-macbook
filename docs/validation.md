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

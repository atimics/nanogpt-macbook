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

# Benchmarks on a CI Mac

Use the [Mac GPU benchmark workflow](https://github.com/cenetex/nanogpt-macbook/actions/workflows/benchmark.yml)
to measure speed away from a busy development Mac. Run **probe** first. Its
`host.json` records Metal access, a small GPU calculation, the runner image,
the chip, and the Python, MLX, and NumPy versions.

The regular CI workflow also runs the test suite on `macos-26` for every push
and pull request. It requires Metal access, so GPU tests run alongside the CPU
tests. The separate Linux job checks portability, lint, formatting, and packaging.

The workflow offers two hosts:

- `macos-26`: the standard Mac runner, free for this public repository. Our
  [September 30 probe](https://github.com/cenetex/nanogpt-macbook/actions/runs/36748164503)
  passed real Metal execution on an Apple Paravirtual device with 7 GiB memory.
  [Host receipt](ci/36748164503/host.json). Use this host first. Each new runner
  image still gets a fresh GPU probe and same-commit noise check.
- `macos-26-xlarge`: GitHub's M2 runner with GPU acceleration. It requires
  GitHub Team or Enterprise Cloud, paid runner access, and a spending budget.
  The published rate on September 30, 2026 is $0.102 per minute. The 30-minute
  job limit bounds runner charges to about $3.06 per run, plus any plan fees.

Sources: [runner specifications](https://docs.github.com/en/actions/reference/runners/larger-runners),
[pricing](https://docs.github.com/en/billing/reference/actions-runner-pricing).

## Run a comparison

1. Select **compare**, a host with a successful Metal probe, and one preset.
2. Enter full baseline and candidate commit hashes. Empty fields use the
   workflow commit, which gives an additional same-commit check.
3. Download the `mac-benchmark` artifact after the run. Start with `summary.md`,
   then inspect the raw receipts and `comparison.json`.

For the latest packed AdamW change, use baseline
`2c7f63f1fc615e4fab7a525faaa1831724752a06` and candidate
`07b11a7b9a8f0bcc883583776178809faca3190e`. Select `medium`, the affected preset.

Each preset first runs six A/A pairs using the baseline on both sides. It then
runs six A/B pairs. The order alternates A/B and B/A. Every sample starts a
fresh process, uses seed 1337, warms up for 20 steps, and measures 100 steps.
It uses float32, a two-step queue, and the existing 2 GiB memory and cache limits.
Each pair runs on the same worker. A fixed lock file keeps dependencies equal.
Receipts verify the loaded commit and the model, training, and host settings.

Compare the A/B gain with the A/A spread before drawing a conclusion. Repeat
small effects across several jobs. A hosted VM is still subject to noise;
calibration measures that noise for each job. The preset's model settings and
all raw step intervals are kept in the artifact. Partial samples survive a
later failure or timeout when the artifact upload step can run.

Runs are manual and serialized within this repository. Choose one preset per
job when the full suite would exceed the 30-minute limit. Artifacts are kept
for 90 days. Preserve accepted results in a results PR before that period ends.

## Publish the results

The current Pages charts contain local M4 Max measurements. Treat small local
speed changes as tentative while we measure them again on a CI Mac. Keep that
history as a record of the original runs.

CI comparisons use a separate receipt format and remain in workflow artifacts
until reviewed. A new chart series needs its own hardware, OS, dependency, and
protocol identity. Compare each commit with a baseline from that same series.
The site builder already enforces one machine and software stack per report.

A dedicated Apple Silicon Mac is another option. Give it a separate runner
label, use one job at a time, and keep other GPU work idle during measurements.
Its access and host setup should be agreed before adding that runner to this
public repository.

## First controlled result: medium training

[Run 36748450305](https://github.com/cenetex/nanogpt-macbook/actions/runs/36748450305)
finished on September 30, 2026 in 10 minutes. It used `macos-26`, image
`20260907.0351.1`, macOS 26.6.2, Python 3.12.10, MLX 0.32.3, and NumPy 2.5.3.
The GPU was an Apple Paravirtual device with 7 GiB exposed memory.

| Comparison | Median paired change | Pair range | Candidate faster |
| --- | ---: | ---: | ---: |
| Same baseline twice (A/A) | +0.04% | -2.74% to +3.66% | 3 of 6 |
| Packed AdamW vs baseline (A/B) | +2.07% | +1.50% to +19.06% | 6 of 6 |

The gain is promising, but it lies within the same-commit variation. The first
A/B pair also has a large timing outlier. More runs are needed to establish a
small speed gain. These measurements support using the runner for remote tests;
a dedicated quiet Mac remains useful for resolving changes below a few percent.

| Source commit | Median throughput in A/B | Peak active MLX memory |
| --- | ---: | ---: |
| [`2c7f63f`](https://github.com/cenetex/nanogpt-macbook/commit/2c7f63f1fc615e4fab7a525faaa1831724752a06) | 11,475 bytes/s | 1,719.868 MiB |
| [`07b11a7`](https://github.com/cenetex/nanogpt-macbook/commit/07b11a7b9a8f0bcc883583776178809faca3190e) | 11,722 bytes/s | 1,719.911 MiB |

This host shows essentially equal peak memory for the two versions. Memory
savings measured on the M4 Max depend on that host and its software stack.
The paired speed change uses the median of the six per-pair ratios, so it can
differ from the ratio of the two median rates above.

All [24 raw samples](ci/36748450305/) are preserved with their source hashes.
The [comparison receipt](ci/36748450305/comparison.json) includes the host,
run order, per-sample file hashes, and all paired ratios. The harness commit is
`925df18710f14b7132e67e28dc3a2b54d9e1cbd6`.

## All presets

All three presets now have a hosted rerun. Each job used the same baseline
`2c7f63f` and candidate `07b11a7`, with six A/A pairs and six A/B pairs on its
own worker. All three reported the same runner image and software versions.

| Preset | Paired change | Same-commit range | Candidate faster | Raw run |
| --- | ---: | ---: | ---: | --- |
| tiny | -7.98% | -16.31% to +6.39% | 2 of 6 | [36750498161](ci/36750498161/) |
| small | +1.17% | -1.35% to +7.63% | 4 of 6 | [36750509324](ci/36750509324/) |
| medium | +2.07% | -2.74% to +3.66% | 6 of 6 | [36748450305](ci/36748450305/) |

Every measured change lies inside that preset's same-commit range. These runs
show why noise checks matter: the free hosted GPU also has variable timing,
especially for short tiny-model samples. Use longer samples or a quiet,
dedicated Apple Silicon host before treating changes of this size as settled.

The Pages site shows separate CI commit comparisons with both baseline and
previous-commit percentages. It calculates chart values from the six A/B samples
per commit. Those percentages compare median rates; the table above uses the
median of paired ratios. Each card links its raw receipts and displays its
same-commit range. The original M4 Max history stays in its own chart section.
All 72 samples are kept in this repository.

## Hosted numerical checks

The first hosted Metal test run passed 374 of 376 tests. Two end-to-end checks
exceeded their per-value tolerance at one token-embedding value each:

- After five AdamW steps, grouped clipping differed from native clipping by
  `2.17e-7` at one weight. A native repeat differed by at most `4.66e-10`.
- Last-token projection differed from full-output projection by `5.48e-6` at
  one near-zero embedding gradient. The largest difference anywhere in that
  gradient array was `1.53e-5`; most values passed the relative tolerance.
  The repeated full-output gradient differed by at most `1.91e-6`.

[Original run](https://github.com/cenetex/nanogpt-macbook/actions/runs/36748837320)
and [repeat controls](https://github.com/cenetex/nanogpt-macbook/actions/runs/36749711785)
retain the full diagnostics. These are repeatable differences between float32
calculation orders. The direct clipping and kernel accuracy tests passed.

The affected end-to-end checks now allow `5e-7` for the embedding weights after
five updates and `1e-5` for GPU embedding gradients in the last-token check.
They also require total embedding error to stay below `2e-6` of the reference
array norm. Native-repeat controls, other parameter values, optimizer state,
and direct kernel checks retain their existing per-value limits. The test logs
show native-repeat and optimized-path errors on each hosted run.

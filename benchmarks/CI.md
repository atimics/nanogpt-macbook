# Benchmarks on a CI Mac

Use the [Mac GPU benchmark workflow](https://github.com/cenetex/nanogpt-macbook/actions/workflows/benchmark.yml)
to measure speed away from a busy development Mac. Run **probe** first. Its
`host.json` records Metal access, a small GPU calculation, the runner image,
the chip, and the Python, MLX, and NumPy versions.

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

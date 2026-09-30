# A 24-hour M2 experiment

Use one AWS `mac2-m2.metal` host in Oregon for one day. Run an AI coding agent
through a remote service. Keep the Mac's GPU available for one benchmark at a
time. Measure the whole edit, check, and benchmark loop.

The initial capacity estimate is **28–42 full iterations per day**, with
**33 iterations** in the middle scenario. These are planning scenarios based
on hosted CI timings. The first AWS M2 session supplies the actual capacity.

## Price and time

The host costs **$0.878/hour**, or **$21.072 for the 24-hour minimum**, in USD.
This is about $21.07 plus AI inference, storage, network, tax, and any extra host
time. A $20 hardware cap falls $1.072 short of the M2 minimum.

Price checked September 30, 2026 against the
[AWS price feed](https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/dedicatedhost-ondemand.json).
The [saved inputs](planning/m2-24h-inputs.json) include the rate code and feed
publication time. AWS applies a [24-hour host minimum](https://aws.amazon.com/ec2/instance-types/mac/faqs/).
The clock starts at host allocation. Host release ends billing. Keep release
confirmation and the final bill with the session record.

Reserve 60 minutes for host readiness, setup, correctness checks, and an initial
comparison. Reserve another 60 minutes for final confirmation, result export,
and cleanup. That leaves **1,320 minutes** for serial iterations. Extra setup,
longer tests, and host release delays change the final capacity and cost.

## What counts as an iteration

One iteration is one candidate commit with a clear hypothesis, an AI edit,
correctness checks, a PR, and a calibrated comparison with the current best
commit. Each preset has six same-commit A/A pairs and six A/B pairs, for
**24 raw samples**. All three presets use **72 samples per iteration**.
Each sample uses a fresh process, 20 warmup steps, and 100 timed training steps.

Record these counts separately:

- **Attempted:** a candidate received agent work. Include failed edits and tests.
- **Benchmarked:** checks passed and every required comparison finished.
- **Accepted:** review and repeat measurements support the claimed improvement.

An API call is one part of agent work. Include all calls, retries, tool waits,
reviews, and repair attempts in its time and inference cost. Capacity estimates
assume every candidate reaches the full benchmark. A real session reports all
three counts above, along with its unfinished work.

## Starting estimate

Our hosted CI comparison steps took 100 seconds for tiny, 276 seconds for small,
and 586 seconds for medium: **962 seconds, or 16.03 minutes**, for the full set.
These times include warmup, process startup, noise checks, comparisons, saved
receipts, and checkout cleanup. Sources and timestamps are in the saved inputs.
They come from an Apple Paravirtual GPU with 7 GiB memory.

Use 10 minutes for agent work and 5 minutes for correctness checks, PR handling,
and related waits. The M2 duration multiplier below is an assumption that the
first dedicated-host calibration will replace.

| Benchmark duration vs hosted CI | Benchmark time | Whole iteration | Iterations/day | Raw samples |
| --- | ---: | ---: | ---: | ---: |
| 1× | 16.03 min | 31.03 min | 42 | 3,024 |
| 1.5× | 24.05 min | 39.05 min | 33 | 2,376 |
| 2× | 32.07 min | 47.07 min | 28 | 2,016 |

The multipliers express planning assumptions. Existing CI controls showed
substantial timing noise. Keep the calibration results beside every
speed claim. Longer samples require a fresh capacity estimate and a matching
comparison protocol.

At 33 iterations, the hardware share is about **$0.64 per iteration**.
Example inference costs, each chosen as a planning allowance:

| AI cost per whole iteration | AI cost for 33 iterations | Host + AI |
| --- | ---: | ---: |
| $0.25 | $8.25 | $29.32 |
| $1.00 | $33.00 | $54.07 |
| $3.00 | $99.00 | $120.07 |

Add storage, network, tax, and extra host time to each total. For an API model,
calculate inference cost from billed uncached input tokens, cached input tokens,
and output tokens at their respective rates. Include billed reasoning tokens
under the provider's output rules. Sum all calls in the iteration. Save the
model ID, provider, usage receipts, rates, and rate date. A subscription agent
can use its own usage and spending records as the accounting basis.

## Replay the calculation

Run this with Python 3.11 or later:

```bash
python scripts/experiment_budget.py --out runs/m2-plan.json
python scripts/experiment_budget.py --slowdown 1 --json
python scripts/experiment_budget.py --slowdown 2 --json
python scripts/experiment_budget.py --agent-minutes 20 --inference-usd-per-iteration 3
python scripts/experiment_budget.py --total-budget-usd 40 --extra-cost-usd 3
```

The last command allows 15 iterations at $1 of inference each. It reserves the
host charge and $3 of extra cost first. `--inference-budget-usd` caps the AI part;
`--hardware-budget-usd` checks the host minimum. These switches calculate a plan.
Session controls enforce time and spending limits when the run is launched.

The count is `floor(available minutes / whole iteration minutes)`, then reduced
to fit the chosen budgets. The JSON records every assumption, price source,
timing source, and count. The default $1 AI allowance is a scenario input.

## Run and measure the process

1. Freeze the starting commit, harness commit, lock file, prompts, agent model,
   and correctness rules. Agree on the host window and inference spending cap.
   Prepare the host setup and release control before allocating the Mac.
2. Start the host clock at allocation. Record its host ID, instance ID, AMI,
   region, macOS, Python, MLX, and NumPy versions. Confirm the Apple M2 Metal
   device with `scripts/ci_benchmark.py probe`. Install the locked environment.
3. Run all presets with the baseline on both sides. The harness now saves total
   wall time in `comparison.json`. Use that file to replace the estimated
   benchmark duration in the planner. Recheck timing as the session progresses.
4. Give the agent one hypothesis and a separate worktree for each candidate.
   Save its prompt and model settings. Commit and push each candidate, and open
   a PR. Keep the benchmark harness and acceptance tests at the frozen revision.
5. Run the correctness checks. For candidates that pass, compare against the
   current best commit on the same host. Keep all samples and failures. Inspect
   memory, loss, numerical checks, and every affected preset as well as speed.
6. Judge the effect against that comparison's same-commit variation. Confirm
   promising changes in another comparison. Include confirmation time in the
   iteration's ledger. Increase the planner's time allowance as measurements
   require. Review the PR and merge accepted changes after the checks pass.
7. Start each next iteration only when the remaining time and inference budget
   cover its allowance. Export the ledger, candidate PRs, and receipts before
   cleanup. Release the host after its minimum allocation window. Verify release
   from outside the Mac and reconcile the bill.

Use full 40-character commit hashes in the commands below. Run the harness from
the frozen checkout. `BASELINE_SHA` is the starting commit for calibration:

```bash
uv run python scripts/ci_benchmark.py compare --preset all \
  --baseline "$BASELINE_SHA" --candidate "$BASELINE_SHA" \
  --out runs/m2-calibration
python scripts/experiment_budget.py \
  --calibration runs/m2-calibration/comparison.json --out runs/m2-measured-plan.json
```

The planner checks completeness, preset coverage, sample hashes, the M2 device,
and a positive wall time. The harness enforces source, environment, workload,
and timing settings during the comparison. Use the largest of several observed
iteration times when reserving room for the next attempt.

## Measure the agent as well as the code

Keep one ledger row per attempt, including failures. Each row needs the session
ID, attempt number, frozen baseline, previous best, candidate hash, PR URL,
hypothesis, model ID, prompt hash, start/end times, agent seconds, check/PR
seconds, benchmark seconds, confirmation seconds, all token counts and costs,
status, rejection reason when relevant, and paths to the raw receipts.

At session end, report attempted, benchmarked, and accepted candidates; completed
iterations per hour; AI and host cost per attempt and per accepted change; and
median and slowest iteration times. A session with zero accepted candidates
reports the counts and total spend, with cost per accepted change left empty.

Use the same final host and frozen protocol to compare the final accepted code
with the starting baseline. Publish throughput gain, memory, correctness, and
learning checks next to total dollars and agent usage. Also compare each accepted
commit with its previous best. Keep AWS M2 measurements in their own series on
the benchmark site. The frozen final check and complete attempt history make
agent sessions comparable across model choices and budgets.

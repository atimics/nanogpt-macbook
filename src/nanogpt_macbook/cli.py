"""The nanogpt command line."""

import argparse
import json
import platform
import sys
from dataclasses import replace
from importlib.resources import files
from pathlib import Path

from . import __version__
from .config import PRESETS, ModelConfig, TrainConfig
from .data import prepare


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="nanoGPT MacBook edition — train a GPT with MLX")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check Python, MLX, and the Metal GPU")
    commands.add_parser("presets", help="Show model sizes")
    measuring = commands.add_parser("benchmark", help="Measure training speed and save raw timings")
    measuring.add_argument("--preset", choices=("all", *PRESETS), default="all")
    measuring.add_argument("--out", type=Path, required=True)
    measuring.add_argument("--steps", type=int, default=100)
    measuring.add_argument("--warmup", type=int, default=20)
    measuring.add_argument("--repeats", type=int, default=3)
    measuring.add_argument("--seed", type=int, default=1337)
    measuring.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    measuring.add_argument("--memory-gb", type=float, default=2)
    measuring.add_argument("--sync", action="store_true", help="Wait after each training step")
    data = commands.add_parser("prepare", help="Turn UTF-8 text into training and validation data")
    source = data.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="A UTF-8 text file")
    source.add_argument("--demo", action="store_true", help="Use the bundled short story")
    data.add_argument("--out", type=Path, required=True)
    data.add_argument("--validation-fraction", type=float, default=0.1)

    training = commands.add_parser("train", help="Train a new GPT")
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--preset", choices=PRESETS, default="tiny")
    training.add_argument("--context", type=int)
    training.add_argument("--batch-size", type=int)
    training.add_argument("--accumulation", type=int)
    training.add_argument("--learning-rate", type=float)
    training.add_argument("--seed", type=int)
    continuing = commands.add_parser("resume", help="Continue from the latest complete checkpoint")
    continuing.add_argument("--data", type=Path, help="Prepared data path if the data moved")
    for command in (training, continuing):
        command.add_argument("--sync", action="store_true", help="Wait after each training step")
        command.add_argument("--run", type=Path, required=True)
        command.add_argument(
            "--steps", type=int, required=True, help="Total target optimizer steps"
        )
        command.add_argument("--eval-every", type=int, default=100)
        command.add_argument("--eval-batches", type=int, default=10)
        command.add_argument("--log-every", type=int, default=10)
        command.add_argument(
            "--time-limit", type=float, help="Stop and save after this many seconds"
        )

    sampling = commands.add_parser("sample", help="Generate a continuation from a saved GPT")
    sampling.add_argument("--run", type=Path, required=True)
    sampling.add_argument("--checkpoint", choices=("best", "latest"), default="best")
    sampling.add_argument("--prompt", default="The ")
    sampling.add_argument("--tokens", type=int, default=200, help="Number of new UTF-8 byte tokens")
    sampling.add_argument("--temperature", type=float, default=0.8, help="Use 0 for greedy output")
    sampling.add_argument(
        "--top-k", type=int, default=40, help="Use 0 to sample the full vocabulary"
    )
    sampling.add_argument("--seed", type=int, default=42)
    scoring = commands.add_parser(
        "evaluate", help="Measure held-out loss in nats and bits per byte"
    )
    scoring.add_argument("--run", type=Path, required=True)
    scoring.add_argument("--data", type=Path)
    scoring.add_argument("--checkpoint", choices=("best", "latest"), default="best")
    scoring.add_argument("--batches", type=int, default=20)
    scoring.add_argument("--batch-size", type=int, default=8)
    for command in (training, continuing, sampling, scoring):
        command.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto")
        command.add_argument(
            "--memory-gb", type=float, default=2, help="MLX GPU memory limit in GiB"
        )
    return root


def doctor() -> int:
    details = {
        "python": platform.python_version(),
        "system": platform.system(),
        "architecture": platform.machine(),
        "macos": platform.mac_ver()[0],
    }
    try:
        import mlx.core as mx

        details["mlx"] = mx.__version__
        details["metal"] = mx.metal.is_available()
        if details["metal"]:
            details["gpu"] = mx.device_info()["device_name"]
        else:
            details["next_step"] = "Use --device cpu for a small check"
    except (ImportError, OSError) as error:
        details["mlx_error"] = str(error)
        details["next_step"] = "Run uv sync with a native Apple Silicon Python on macOS 14 or later"
        print(json.dumps(details, indent=2))
        return 1
    print(json.dumps(details, indent=2))
    return 0


def dispatch(args):
    if args.command == "benchmark":
        from .benchmark import benchmark

        benchmark(
            args.out,
            list(PRESETS) if args.preset == "all" else [args.preset],
            device=args.device,
            memory_gb=args.memory_gb,
            steps=args.steps,
            warmup=args.warmup,
            repeats=args.repeats,
            seed=args.seed,
            pipeline=False if args.sync else None,
        )
        return 0
    if args.command == "doctor":
        return doctor()
    if args.command == "presets":
        print("Preset   Parameters   Layers   Width   Context   Batch")
        for name, (model, training) in PRESETS.items():
            width = model.width
            count = model.layers * (12 * width**2 + 4 * width)
            count += (model.vocab_size + model.context + 2) * width
            print(
                f"{name:<8} {count:>10,} {model.layers:>8} {width:>7} "
                f"{model.context:>9} {training.batch_size:>7}"
            )
        return 0
    if args.command == "prepare":
        source = (
            Path(str(files("nanogpt_macbook").joinpath("demo.txt"))) if args.demo else args.input
        )
        print(json.dumps(prepare(source, args.out, args.validation_fraction), indent=2))
        return 0

    from . import checkpoint
    from .data import Dataset
    from .engine import evaluate, generate, load_model, select_device, train

    select_device(args.device, args.memory_gb)
    if args.command in ("train", "resume"):
        if args.command == "train":
            model, training = PRESETS[args.preset]
            if args.context is not None:
                model = replace(model, context=args.context)
            overrides = {
                key: getattr(args, key)
                for key in ("batch_size", "accumulation", "learning_rate", "seed")
                if getattr(args, key) is not None
            }
            if "learning_rate" in overrides:
                overrides["min_learning_rate"] = overrides["learning_rate"] / 10
            training = replace(training, **overrides)
            data = args.data
        else:
            _, state = checkpoint.read(args.run)
            data = args.data or Path(state["data_path"])
            model = ModelConfig(**state["model_config"])
            training = TrainConfig(**state["train_config"])
        train(
            data,
            args.run,
            model,
            training,
            args.steps,
            resume=args.command == "resume",
            eval_every=args.eval_every,
            eval_batches=args.eval_batches,
            log_every=args.log_every,
            time_limit=args.time_limit,
            pipeline=False if args.sync else None,
        )
    elif args.command == "sample":
        model, _ = load_model(args.run, args.checkpoint)
        print(generate(model, args.prompt, args.tokens, args.temperature, args.top_k, args.seed))
    elif args.command == "evaluate":
        model, state = load_model(args.run, args.checkpoint)
        data = Dataset(args.data or Path(state["data_path"]), model.config.context)
        print(
            json.dumps(
                {
                    "step": state["step"],
                    "same_training_data": data.identity == state["dataset"],
                    **evaluate(model, data, args.batch_size, args.batches),
                },
                indent=2,
            )
        )
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        status = dispatch(args)
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"nanogpt: {error}", file=sys.stderr)
        status = 1
    raise SystemExit(status)

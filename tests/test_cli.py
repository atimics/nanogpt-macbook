import os
import signal
import subprocess
import sys
import time


def cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "nanogpt_macbook", *map(str, args)],
        capture_output=True,
        text=True,
        timeout=45,
    )


def test_cli_prepare_train_resume_sample(tmp_path):
    data = tmp_path / "data"
    run = tmp_path / "run"
    assert cli("--help").returncode == 0
    assert cli("prepare", "--demo", "--out", data).returncode == 0
    result = cli(
        "train",
        "--data",
        data,
        "--run",
        run,
        "--steps",
        1,
        "--context",
        8,
        "--batch-size",
        1,
        "--eval-batches",
        1,
        "--device",
        "cpu",
    )
    assert result.returncode == 0, result.stderr
    result = cli("resume", "--run", run, "--steps", 2, "--eval-batches", 1, "--device", "cpu")
    assert result.returncode == 0, result.stderr
    result = cli("sample", "--run", run, "--tokens", 4, "--prompt", "Mira ", "--device", "cpu")
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("Mira ")
    result = cli("evaluate", "--run", run, "--batches", 1, "--device", "cpu")
    assert result.returncode == 0, result.stderr
    assert '"bits_per_byte"' in result.stdout
    result = cli("sample", "--run", run, "--tokens", 0, "--device", "cpu")
    assert result.returncode == 1
    assert "tokens must be positive" in result.stderr


def test_ctrl_c_saves_a_completed_step(tmp_path):
    data, run = tmp_path / "data", tmp_path / "run"
    assert cli("prepare", "--demo", "--out", data).returncode == 0
    log = tmp_path / "log.txt"
    with log.open("w") as stream:
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "nanogpt_macbook",
                "train",
                "--data",
                str(data),
                "--run",
                str(run),
                "--steps",
                "10000",
                "--context",
                "8",
                "--batch-size",
                "1",
                "--eval-batches",
                "1",
                "--log-every",
                "1",
                "--device",
                "cpu",
            ],
            stdout=stream,
            stderr=stream,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and process.poll() is None:
                if "| train " in log.read_text():
                    break
                time.sleep(0.05)
            assert "| train " in log.read_text(), log.read_text()
            process.send_signal(signal.SIGINT)
            assert process.wait(timeout=20) == 0, log.read_text()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    assert "Saved step" in log.read_text()
    result = cli("sample", "--run", run, "--checkpoint", "latest", "--tokens", 4, "--device", "cpu")
    assert result.returncode == 0, result.stderr

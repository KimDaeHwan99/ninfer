#!/usr/bin/env python3
"""Measure the VRAM a running ninfer-serve leaves unused under load and derive --vram-headroom-mib.

`--kv-capacity auto` gives the KV pool everything above the headroom. Memory that stays free on the
fullest serving GPU while the server runs its heaviest paths is headroom that could hold KV
instead. This tool samples nvidia-smi on the serving GPUs while it drives load against an
already-running server: concurrent copy edits that run wide n-gram verify rounds, and TTFT runner
cases with long prompts, heterogeneous concurrent requests and heavy media. It then reports the
minimum free memory per GPU and the headroom that would leave `--safety-mib` of it unused.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bench.ttft.cases import CASES  # noqa: E402
from tools.bench.ttft.corpus import DEFAULT_MANIFEST  # noqa: E402

RUNNER = REPO_ROOT / "tools/bench/run_serve_ttft.py"
DEFAULT_CASES = (
    "cold-long-64k",
    "mixed-four-concurrent",
    "many-image-28",
    "two-heavy-media-arrivals",
)
DEFAULT_COPY_SOURCE = REPO_ROOT / "src/ops/kernel/tensor_parallel.cuh"


def media_cases(manifest: Path = DEFAULT_MANIFEST) -> frozenset[str]:
    """Cases that send at least one image or video, which a Serve without --vision rejects."""
    media = json.loads(manifest.read_text(encoding="utf-8"))["media"]

    def is_media(corpus_id: str) -> bool:
        return corpus_id in media or corpus_id.startswith("load-image")

    return frozenset(
        name for name, case in CASES.items() if any(is_media(c) for c in case.corpus_ids)
    )


MEDIA_CASES = media_cases()


@dataclass(frozen=True)
class Sample:
    seconds: float
    gpu: int
    free_mib: int
    used_mib: int
    total_mib: int


def parse_sample(line: str, seconds: float) -> Sample | None:
    """Parse one `index, memory.free, memory.used, memory.total` nvidia-smi CSV line."""
    fields = [part.strip() for part in line.split(",")]
    if len(fields) != 4:
        return None
    try:
        gpu, free, used, total = (int(value) for value in fields)
    except ValueError:
        return None
    return Sample(seconds, gpu, free, used, total)


@dataclass
class Recommendation:
    min_free_mib: dict[int, int]
    fullest_gpu: int
    slack_mib: int
    headroom_mib: int | None  # None when the samples contradict the given headroom
    explanation: str


def recommend(
    min_free_mib: dict[int, int], current_headroom_mib: int, safety_mib: int
) -> Recommendation:
    """Move the free memory above `safety_mib` on the fullest GPU from the headroom into KV."""
    if not min_free_mib:
        raise ValueError("no GPU memory samples were collected")
    fullest = min(min_free_mib, key=lambda gpu: min_free_mib[gpu])
    least = min_free_mib[fullest]
    slack = least - safety_mib
    if slack > current_headroom_mib:
        # --kv-capacity auto leaves only the headroom free, so more than that means the server
        # runs another headroom, a fixed --kv-capacity, or on other cards.
        return Recommendation(
            dict(min_free_mib), fullest, slack, None,
            f"GPU {fullest} never had less than {least} MiB free, more than the "
            f"{current_headroom_mib} MiB headroom plus the {safety_mib} MiB margin. Check that "
            "--headroom-mib matches the server, that it runs --kv-capacity auto, and that --gpus "
            "names its cards (compare the startup 'capacity | ... | free' line).",
        )
    headroom = current_headroom_mib - slack
    if slack >= 0:
        explanation = (
            f"GPU {fullest} never had less than {least} MiB free; keeping {safety_mib} MiB of "
            f"that unused lets the KV pool take the other {slack} MiB."
        )
    else:
        explanation = (
            f"GPU {fullest} fell to {least} MiB free, below the {safety_mib} MiB safety margin; "
            f"raise the headroom by {-slack} MiB."
        )
    return Recommendation(dict(min_free_mib), fullest, slack, headroom, explanation)


@dataclass
class Sampler:
    gpus: Sequence[int]
    interval_ms: int
    samples: list[Sample] = field(default_factory=list)
    _process: subprocess.Popen[str] | None = None
    _thread: threading.Thread | None = None
    _started: float = 0.0

    def start(self) -> None:
        command = [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.used,memory.total",
            "--format=csv,noheader,nounits",
            "-i",
            ",".join(str(gpu) for gpu in self.gpus),
            "-lms",
            str(self.interval_ms),
        ]
        self._started = time.monotonic()
        self._process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        self._thread = threading.Thread(target=self._read, name="nvidia-smi", daemon=True)
        self._thread.start()

    def _read(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            sample = parse_sample(line, time.monotonic() - self._started)
            if sample is not None:
                self.samples.append(sample)

    def _missing(self) -> list[int]:
        return sorted(set(self.gpus) - {sample.gpu for sample in self.samples})

    def require_running(self) -> None:
        """Fail before any load is sent unless nvidia-smi is sampling every requested GPU."""
        assert self._process is not None
        missing = self._missing()
        if self._process.poll() is None and not missing:
            return
        self.stop()
        raise RuntimeError(f"nvidia-smi is not sampling GPU(s) {missing}")

    def stop(self) -> None:
        """Stop sampling; raise if nvidia-smi failed during the run or skipped a GPU."""
        if self._process is None:
            return
        process, self._process = self._process, None
        exited = process.poll()
        if exited is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if self._thread is not None:
            self._thread.join(timeout=5)
        error = process.stderr.read().strip() if process.stderr is not None else ""
        if exited not in (None, 0):
            raise RuntimeError(f"nvidia-smi exited with status {exited}: {error}")
        missing = self._missing()
        if missing:
            raise RuntimeError(f"nvidia-smi reported no samples for GPU(s) {missing}: {error}")

    def min_free_mib(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for sample in self.samples:
            out[sample.gpu] = min(out.get(sample.gpu, sample.free_mib), sample.free_mib)
        return out


def request(url: str, api_key: str | None, body: object | None, timeout: float) -> object:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = None if body is None else json.dumps(body).encode("utf-8")
    with urllib.request.urlopen(
        urllib.request.Request(url, data=data, headers=headers), timeout=timeout
    ) as response:
        return json.load(response)


def copy_edit(args: argparse.Namespace, model: str, source: str) -> str:
    """One greedy request that reproduces `source`, so n-gram copy proposals verify wide."""
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Reproduce the following file exactly, changing nothing. Output only "
                "the file.\n\n```\n" + source + "\n```",
            }
        ],
        "max_tokens": args.copy_max_tokens,
        "temperature": 0,
        "enable_thinking": False,
    }
    try:
        payload = request(
            f"{args.base_url}/v1/chat/completions", args.api_key, body, args.timeout_seconds
        )
        return payload["choices"][0].get("finish_reason") or "ok"  # type: ignore[index]
    except Exception as error:  # Load generation only: report and keep sampling.
        return f"failed: {type(error).__name__}: {error}"


def run_copy_burst(args: argparse.Namespace, model: str, source: str) -> list[str]:
    with ThreadPoolExecutor(max_workers=args.copy_requests) as pool:
        futures = [
            pool.submit(copy_edit, args, model, source) for _ in range(args.copy_requests)
        ]
        return [future.result() for future in futures]


def run_case(args: argparse.Namespace, name: str) -> int:
    """Run one TTFT runner case as load. The runner executes a case only under its expected
    profile label, so the artifacts carry that label although the server runs another profile;
    they are load generation, not TTFT evidence."""
    cases = args.output / "cases"
    cases.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(RUNNER),
        "--base-url",
        args.base_url,
        "--case",
        name,
        "--profile-label",
        CASES[name].profile,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--api-key-env",
        args.api_key_env,
        "--output",
        str(cases / f"{name}.json"),
    ]
    with (cases / f"{name}.stdout.json").open("w", encoding="utf-8") as stdout:
        return subprocess.run(command, cwd=REPO_ROOT, stdout=stdout, check=False).returncode


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", required=True, help="running ninfer-serve, e.g. http://127.0.0.1:8080"
    )
    parser.add_argument("--api-key-env", default="NINFER_API_KEY")
    parser.add_argument("--gpus", default="0,1", help="nvidia-smi indices of the serving GPUs")
    parser.add_argument(
        "--headroom-mib", type=int, default=1024, help="--vram-headroom-mib the server runs with"
    )
    parser.add_argument(
        "--safety-mib", type=int, default=256, help="free memory to keep on the fullest GPU"
    )
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES), help="TTFT runner cases")
    parser.add_argument(
        "--no-media", action="store_true", help="skip cases that send images or video"
    )
    parser.add_argument("--copy-requests", type=int, default=4, help="concurrent copy edits")
    parser.add_argument("--copy-max-tokens", type=int, default=1536)
    parser.add_argument("--copy-source", type=Path, default=DEFAULT_COPY_SOURCE)
    parser.add_argument("--interval-ms", type=int, default=100, help="nvidia-smi sample period")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--output", type=Path, required=True, help="directory for samples/report")
    args = parser.parse_args(argv)
    args.base_url = args.base_url.rstrip("/")
    args.api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    gpus = [int(value) for value in args.gpus.split(",") if value]
    cases = [name for name in args.cases.split(",") if name]
    unknown = [name for name in cases if name not in CASES]
    if unknown:
        parser.error("unknown case(s): " + ", ".join(unknown))
    if args.no_media:
        cases = [name for name in cases if name not in MEDIA_CASES]
    if args.copy_requests < 0 or args.safety_mib < 0 or args.headroom_mib < 0:
        parser.error("--copy-requests, --safety-mib and --headroom-mib must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.copy_source.read_text(encoding="utf-8")
    models = request(f"{args.base_url}/v1/models", args.api_key, None, args.timeout_seconds)
    model = models["data"][0]["id"]  # type: ignore[index]

    sampler = Sampler(gpus, args.interval_ms)
    sampler.start()
    steps: list[dict[str, object]] = []
    try:
        time.sleep(1.0)
        sampler.require_running()
        idle = sampler.min_free_mib()
        if args.copy_requests:
            steps.append({"step": "copy edits", "results": run_copy_burst(args, model, source)})
        for name in cases:
            print(f"[headroom] case {name}", file=sys.stderr, flush=True)
            steps.append({"step": name, "exit_code": run_case(args, name)})
        if args.copy_requests:
            steps.append({"step": "copy edits", "results": run_copy_burst(args, model, source)})
    finally:
        sampler.stop()

    result = recommend(sampler.min_free_mib(), args.headroom_mib, args.safety_mib)
    failed = [
        step
        for step in steps
        if step.get("exit_code") not in (None, 0)
        or any(str(outcome).startswith("failed") for outcome in step.get("results", []))
    ]
    with (args.output / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seconds", "gpu", "free_mib", "used_mib", "total_mib"])
        for sample in sampler.samples:
            writer.writerow(
                [f"{sample.seconds:.3f}", sample.gpu, sample.free_mib, sample.used_mib,
                 sample.total_mib]
            )
    report = {
        "base_url": args.base_url,
        "model": model,
        "gpus": gpus,
        "current_headroom_mib": args.headroom_mib,
        "safety_mib": args.safety_mib,
        "idle_free_mib": idle,
        "min_free_mib": result.min_free_mib,
        "fullest_gpu": result.fullest_gpu,
        "slack_mib": result.slack_mib,
        "recommended_headroom_mib": result.headroom_mib,
        "explanation": result.explanation,
        "complete": not failed,
        "steps": steps,
        "samples": len(sampler.samples),
        "note": "cases/* are load generation under the serving profile, not TTFT evidence; "
        "their profile labels only satisfy the runner.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for gpu in sorted(result.min_free_mib):
        print(f"GPU {gpu}: idle free {idle.get(gpu, '?')} MiB, minimum free under load "
              f"{result.min_free_mib[gpu]} MiB")
    print(result.explanation)
    if result.headroom_mib is not None:
        print(f"Recommended: --vram-headroom-mib {result.headroom_mib}")
    if failed:
        # A case that exits 2 did not construct its full request graph, so it may have loaded the
        # server less than designed.
        print("Some load steps failed or ran short; check them (cases/<name>, report.json) "
              "before trusting the minimum: " + ", ".join(str(step["step"]) for step in failed))
        return 2
    return 0 if result.headroom_mib is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())

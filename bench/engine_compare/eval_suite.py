#!/usr/bin/env python3
"""Evaluate fixed benchmark subsets against an OpenAI-compatible chat server.

Requires Python 3.14 and pyarrow. HumanEval additionally requires POSIX resource
limits and a python3 executable.

SECURITY: HumanEval executes untrusted generated Python. The requested resource
limits are not a security boundary: they do not isolate files, networking, or
other processes. Run this script inside a disposable, externally isolated
container/VM without secrets when enabling HumanEval.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import csv
import dataclasses
import datetime
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request


SEED = 20261023
SUITES = ("gsm8k", "mmlupro", "humaneval", "kmmlu")
REQUEST_TIMEOUT = 1800
HUMANEVAL_TIMEOUT = 10
SCHEMA_VERSION = 1

NUMBER_PATTERN = (
    r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
NUMBER_RE = re.compile(NUMBER_PATTERN)
ANSWER_NUMBER_RE = re.compile(
    rf"(?i:\bANSWER):\s*\$?\s*({NUMBER_PATTERN})"
)
PYTHON_BLOCK_RE = re.compile(
    r"```[ \t]*python[ \t]*\r?\n(.*?)```", re.IGNORECASE | re.DOTALL
)


@dataclasses.dataclass(frozen=True)
class Item:
    suite: str
    id: str
    prompt: str
    gold: str = ""
    category: str | None = None
    original_prompt: str = ""
    test: str = ""
    entry_point: str = ""


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def item_key(item: Item) -> tuple[str, str]:
    return item.suite, item.id


def required_text(row: dict, name: str, location: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: missing or invalid {name!r}")
    return value


def read_parquet(path: Path, columns: list[str]) -> list[dict]:
    # Imported lazily: the isolated HumanEval helper runs with -S.
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Install pyarrow to read the benchmark parquet files") from exc
    return pq.read_table(path, columns=columns).to_pylist()


def sample_indices(count: int, size: int, rng: random.Random, label: str) -> list[int]:
    if count < size:
        raise ValueError(f"{label}: need {size} rows, found {count}")
    # Row indices are the stable IDs for datasets without an explicit ID.
    return rng.sample(list(range(count)), size)


def load_gsm8k(data: Path) -> list[Item]:
    rows = read_parquet(data / "gsm8k-test.parquet", ["question", "answer"])
    rng = random.Random(SEED)
    items = []
    for index in sample_indices(len(rows), 100, rng, "gsm8k"):
        row = rows[index]
        location = f"gsm8k row {index}"
        question = required_text(row, "question", location)
        answer = required_text(row, "answer", location)
        if "####" not in answer:
            raise ValueError(f"{location}: gold answer has no #### delimiter")
        gold = answer.rsplit("####", 1)[1].strip().replace(",", "")
        if not NUMBER_RE.fullmatch(gold) or not math.isfinite(float(gold)):
            raise ValueError(f"{location}: invalid numeric gold answer {gold!r}")
        items.append(
            Item(
                suite="gsm8k",
                id=str(index),
                prompt=question
                + "\n\nSolve step by step. End with a final line: ANSWER: <number>",
                gold=gold,
            )
        )
    return items


def load_mmlupro(data: Path) -> list[Item]:
    rows = read_parquet(
        data / "mmlu-pro-test.parquet",
        ["question", "options", "answer", "category"],
    )
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        category = required_text(row, "category", f"mmlupro row {index}")
        groups.setdefault(category, []).append(index)

    if len(groups) != 14:
        raise ValueError(f"mmlupro: expected 14 categories, found {len(groups)}")

    rng = random.Random(SEED)
    items = []
    for category in sorted(groups):
        indices = sorted(groups[category])
        if len(indices) < 10:
            raise ValueError(f"mmlupro category {category!r}: fewer than 10 rows")
        for index in rng.sample(indices, 10):
            row = rows[index]
            location = f"mmlupro row {index}"
            question = required_text(row, "question", location)
            options = row.get("options")
            if (
                not isinstance(options, list)
                or not 2 <= len(options) <= 10
                or not all(isinstance(option, str) for option in options)
            ):
                raise ValueError(f"{location}: expected 2–10 string options")
            gold = required_text(row, "answer", location).strip()
            if len(gold) != 1 or gold not in "ABCDEFGHIJ"[: len(options)]:
                raise ValueError(f"{location}: invalid answer letter {gold!r}")
            choices = "\n".join(
                f"{chr(65 + i)}. {option}" for i, option in enumerate(options)
            )
            items.append(
                Item(
                    suite="mmlupro",
                    id=str(index),
                    category=category,
                    gold=gold,
                    prompt=question
                    + "\n\n"
                    + choices
                    + "\n\nAnswer with the letter of the correct option. "
                    "End with a final line: ANSWER: <letter>",
                )
            )
    return items


def load_humaneval(data: Path) -> list[Item]:
    rows = read_parquet(
        data / "humaneval.parquet", ["task_id", "prompt", "test", "entry_point"]
    )
    indexed = []
    seen = set()
    for row in rows:
        task_id = required_text(row, "task_id", "humaneval")
        match = re.fullmatch(r"HumanEval/(\d+)", task_id)
        if not match or task_id in seen:
            raise ValueError(f"humaneval: invalid or duplicate task_id {task_id!r}")
        seen.add(task_id)
        indexed.append((int(match.group(1)), row))

    indexed.sort(key=lambda pair: pair[0])
    if [index for index, _ in indexed] != list(range(164)):
        raise ValueError("humaneval: expected the 164 tasks HumanEval/0 through /163")

    items = []
    # Numeric task order, not lexical order such as /1, /10, /100.
    for _, row in indexed[::2]:
        task_id = row["task_id"]
        prompt = required_text(row, "prompt", task_id)
        test = required_text(row, "test", task_id)
        entry_point = required_text(row, "entry_point", task_id)
        if not entry_point.isidentifier():
            raise ValueError(f"{task_id}: invalid entry_point")
        items.append(
            Item(
                suite="humaneval",
                id=task_id,
                prompt="Complete the following Python function. Return the full "
                "function in one ```python code block.\n\n" + prompt,
                original_prompt=prompt,
                test=test,
                entry_point=entry_point,
            )
        )
    return items


def load_kmmlu(data: Path) -> list[Item]:
    paths = sorted(data.glob("kmmlu-*.csv"), key=lambda path: path.name)
    # The specified 13,13,13,13,12,12,12,12 allocation requires eight files.
    if len(paths) != 8:
        raise ValueError(
            "kmmlu: expected exactly 8 kmmlu-*.csv files for the specified "
            f"100-item allocation; found {len(paths)}"
        )

    rng = random.Random(SEED)
    items = []
    required_columns = {"question", "answer", "A", "B", "C", "D", "Category"}
    for file_index, path in enumerate(paths):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not required_columns.issubset(reader.fieldnames or []):
                raise ValueError(f"{path}: missing required CSV columns")
            rows = list(reader)

        size = 13 if file_index < 4 else 12
        for index in sample_indices(len(rows), size, rng, path.name):
            row = rows[index]
            location = f"{path.name} row {index}"
            question = required_text(row, "question", location)
            category = required_text(row, "Category", location)
            answer = required_text(row, "answer", location).strip()
            if answer not in ("1", "2", "3", "4"):
                raise ValueError(f"{location}: answer must be 1, 2, 3, or 4")
            choices = "\n".join(
                f"{letter}. {required_text(row, letter, location)}"
                for letter in "ABCD"
            )
            items.append(
                Item(
                    suite="kmmlu",
                    id=f"{path.name}:{index}",
                    category=category,
                    gold="ABCD"[int(answer) - 1],
                    prompt=question
                    + "\n\n"
                    + choices
                    + "\n\n정답 보기의 알파벳 하나를 고르세요. "
                    "마지막 줄에 'ANSWER: <알파벳>' 형식으로 답하세요.",
                )
            )
    return items


def load_items(data: Path, suites: list[str], limit: int | None) -> list[Item]:
    loaders = {
        "gsm8k": load_gsm8k,
        "mmlupro": load_mmlupro,
        "humaneval": load_humaneval,
        "kmmlu": load_kmmlu,
    }
    items = []
    for suite in suites:
        # Each loader owns its RNG; enabling another suite cannot change a subset.
        selected = loaders[suite](data)
        if limit is not None:
            selected = selected[:limit]
        items.extend(selected)
    keys = [item_key(item) for item in items]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate benchmark item IDs")
    return items


def extract_number(text: str) -> str:
    matches = ANSWER_NUMBER_RE.findall(text)
    if not matches:
        matches = NUMBER_RE.findall(text)
    return matches[-1].replace(",", "") if matches else ""


def extract_letter(text: str, letters: str) -> str:
    matches = re.findall(rf"(?i:\bANSWER):\s*([{letters}])\b", text)
    if not matches:
        matches = re.findall(rf"(?<!\w)([{letters}])(?!\w)", text[-200:])
    return matches[-1] if matches else ""


def extract_code(text: str, item: Item) -> str:
    blocks = PYTHON_BLOCK_RE.findall(text)
    code = (blocks[-1] if blocks else text).strip("\r\n")
    try:
        tree = ast.parse(code)
        defines_entry = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == item.entry_point
            for node in tree.body
        )
    except (SyntaxError, ValueError):
        # Do not prepend the prompt to a full function that is merely truncated.
        defines_entry = bool(
            re.search(
                rf"(?m)^(?:async[ \t]+)?def[ \t]+"
                rf"{re.escape(item.entry_point)}[ \t]*\(",
                code,
            )
        )
    if not defines_entry:
        prefix = item.original_prompt
        if not prefix.endswith("\n"):
            prefix += "\n"
        code = prefix + code
    return code


def limited_environment() -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", os.defpath),
        "LANG": os.environ.get("LANG") or "C.UTF-8",
    }


def check_humaneval_support() -> str:
    if os.name != "posix":
        raise RuntimeError(
            "HumanEval requires POSIX resource limits; use Linux or another "
            "supported POSIX environment, or omit humaneval"
        )
    import resource

    for name in (
        "RLIMIT_AS",
        "RLIMIT_CPU",
        "RLIMIT_NPROC",
        "RLIMIT_FSIZE",
        "RLIMIT_NOFILE",
    ):
        if not hasattr(resource, name):
            raise RuntimeError(f"HumanEval requires resource.{name}")
    python3 = shutil.which("python3", path=limited_environment()["PATH"])
    if not python3:
        raise RuntimeError("HumanEval requires python3 on PATH")
    return os.path.abspath(python3)


def apply_resource_limits() -> None:
    import resource

    for name, value in (
        ("RLIMIT_AS", 1024**3),
        ("RLIMIT_CPU", 10),
        ("RLIMIT_NPROC", 64),
        ("RLIMIT_FSIZE", 10 * 1024**2),
        ("RLIMIT_NOFILE", 64),
    ):
        resource.setrlimit(getattr(resource, name), (value, value))


def kill_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def file_tail(handle, size: int = 4000) -> str:
    handle.seek(0, os.SEEK_END)
    end = handle.tell()
    handle.seek(max(0, end - size))
    return handle.read(size).decode("utf-8", errors="replace").strip()


def humaneval_helper(python3: str) -> int:
    """Single-threaded helper: preexec_fn must not run in the threaded evaluator."""
    process = None
    outcome = {"correct": False}
    try:
        with open("stdout.txt", "w+b") as stdout_file, open(
            "stderr.txt", "w+b"
        ) as stderr_file:
            process = subprocess.Popen(
                [python3, "-I", "-S", "program.py"],
                cwd=os.getcwd(),
                env=limited_environment(),
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                preexec_fn=apply_resource_limits,
                start_new_session=True,
                close_fds=True,
            )
            timed_out = False
            try:
                returncode = process.wait(timeout=HUMANEVAL_TIMEOUT)
            except subprocess.TimeoutExpired:
                timed_out = True
                kill_process_group(process.pid)
                returncode = process.wait()
            finally:
                # Also remove ordinary descendants left behind by a finished task.
                kill_process_group(process.pid)

            outcome["correct"] = not timed_out and returncode == 0
            if timed_out:
                outcome["error"] = "HumanEval execution timed out after 10 seconds"
            elif returncode != 0:
                detail = file_tail(stderr_file) or file_tail(stdout_file)
                outcome["error"] = f"HumanEval exited with code {returncode}"
                if detail:
                    outcome["error"] += ": " + detail
    except Exception as exc:
        if process is not None:
            try:
                kill_process_group(process.pid)
                process.wait()
            except Exception:
                pass
        outcome = {
            "correct": False,
            "error": f"HumanEval runner failed: {type(exc).__name__}: {exc}",
        }

    print(json.dumps(outcome, ensure_ascii=True, allow_nan=False))
    return 0


def execute_humaneval(
    code: str, item: Item, python3: str
) -> tuple[bool, str | None]:
    # TemporaryDirectory honors TMPDIR and removes the program and outputs.
    with tempfile.TemporaryDirectory(prefix="eval-humaneval-") as directory:
        program = code + "\n\n" + item.test + f"\n\ncheck({item.entry_point})\n"
        Path(directory, "program.py").write_text(program, encoding="utf-8")
        try:
            # Fresh interpreter avoids preexec_fn in a process containing threads.
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    str(Path(__file__).resolve()),
                    "--_humaneval-helper",
                    python3,
                ],
                cwd=directory,
                env=limited_environment(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, "HumanEval helper timed out after 30 seconds"
        except OSError as exc:
            return False, f"Could not start HumanEval helper: {exc}"

        if completed.returncode != 0:
            detail = completed.stderr[-4000:].decode("utf-8", errors="replace")
            return False, f"HumanEval helper exited {completed.returncode}: {detail}"
        try:
            outcome = json.loads(completed.stdout)
            if not isinstance(outcome, dict) or type(outcome.get("correct")) is not bool:
                raise ValueError("invalid helper outcome")
            error = outcome.get("error")
            if error is not None and not isinstance(error, str):
                raise ValueError("invalid helper error")
            return outcome["correct"], error
        except (ValueError, UnicodeError) as exc:
            return False, f"Invalid HumanEval helper response: {exc}"


def request_completion(
    endpoint: str, body: dict
) -> tuple[dict | None, float, str | None]:
    payload = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
    started = time.perf_counter()
    for attempt in range(3):  # Initial attempt plus two connection-error retries.
        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                raw = response.read()
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise ValueError("response JSON is not an object")
            return result, time.perf_counter() - started, None
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read(4000).decode("utf-8", errors="replace")
            except Exception:
                detail = str(exc)
            finally:
                exc.close()
            return (
                None,
                time.perf_counter() - started,
                f"HTTP {exc.code}: {detail}",
            )
        except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
            if attempt < 2:
                time.sleep(2**attempt)
                continue
            return (
                None,
                time.perf_counter() - started,
                f"Connection error after 3 attempts: {type(exc).__name__}: {exc}",
            )
        except (ValueError, UnicodeError) as exc:
            return (
                None,
                time.perf_counter() - started,
                f"Invalid response JSON: {exc}",
            )
    raise AssertionError("unreachable")


def evaluate(item: Item, args: argparse.Namespace, python3: str | None) -> dict:
    record = {
        "suite": item.suite,
        "id": item.id,
        "correct": False,
        "extracted_answer": "",
        "completion_tokens": None,
        "seconds": 0.0,
        "finish_reason": None,
    }
    if item.category is not None:
        record["category"] = item.category

    try:
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": item.prompt}],
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "seed": 7,
            "max_tokens": args.max_tokens,
        }
        if args.budget_field is not None:
            body[args.budget_field] = args.budget

        response, seconds, error = request_completion(
            args.url + "/v1/chat/completions", body
        )
        # Latency includes HTTP retries/backoff, but excludes local grading.
        record["seconds"] = seconds
        if error:
            record["error"] = error
            return record

        usage = response.get("usage")
        if isinstance(usage, dict):
            tokens = usage.get("completion_tokens")
            if type(tokens) is int and tokens >= 0:
                record["completion_tokens"] = tokens

        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError("response has no valid choices[0]")
        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise ValueError("finish_reason is not a string or null")
        record["finish_reason"] = finish_reason

        message = choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("choices[0].message is missing or invalid")
        # Deliberately never fall back to reasoning_content or reasoning.
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("message.content is not a string")

        if item.suite == "gsm8k":
            extracted = extract_number(content)
            record["extracted_answer"] = extracted
            if not extracted:
                record["error"] = "No numeric answer found"
            else:
                value = float(extracted)
                record["correct"] = math.isfinite(value) and math.isclose(
                    value, float(item.gold), rel_tol=0.0, abs_tol=1e-6
                )
        elif item.suite in ("mmlupro", "kmmlu"):
            letters = "A-J" if item.suite == "mmlupro" else "A-D"
            extracted = extract_letter(content, letters)
            record["extracted_answer"] = extracted
            record["correct"] = extracted == item.gold
            if not extracted:
                record["error"] = "No answer letter found"
        else:
            code = extract_code(content, item)
            record["extracted_answer"] = code
            correct, error = execute_humaneval(code, item, python3)
            record["correct"] = correct
            if error:
                record["error"] = error
    except Exception as exc:
        record["correct"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def accuracy_counts(records: list[dict]) -> dict:
    correct = sum(record["correct"] for record in records)
    total = len(records)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
    }


def make_summary(items: list[Item], records: dict, suites: list[str]) -> dict:
    summary = {}
    for suite in suites:
        suite_items = [item for item in items if item.suite == suite]
        completed = [
            records[item_key(item)]
            for item in suite_items
            if item_key(item) in records
        ]
        tokens = [
            record["completion_tokens"]
            for record in completed
            if record["completion_tokens"] is not None
        ]
        metrics = accuracy_counts(completed)
        metrics.update(
            {
                "expected_total": len(suite_items),
                # Missing usage is unknown, not zero.
                "mean_completion_tokens": sum(tokens) / len(tokens) if tokens else None,
                "completion_tokens_reported": len(tokens),
                "mean_seconds": (
                    sum(record["seconds"] for record in completed) / len(completed)
                    if completed
                    else None
                ),
                "cut_off": sum(
                    record["finish_reason"] == "length" for record in completed
                ),
                # Includes extraction failures and HumanEval execution failures.
                "errors": sum(bool(record.get("error")) for record in completed),
            }
        )
        if suite in ("mmlupro", "kmmlu"):
            metrics["per_category"] = {
                category: accuracy_counts(
                    [record for record in completed if record["category"] == category]
                )
                for category in sorted({item.category for item in suite_items})
            }
        summary[suite] = metrics
    return summary


def atomic_write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def load_existing(path: Path, settings: dict, items: list[Item]) -> tuple[dict, dict]:
    if not path.exists():
        return {}, settings

    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Existing output has an unsupported schema")
    previous = document.get("settings")
    if not isinstance(previous, dict):
        raise ValueError("Existing output has no valid run settings")

    # Concurrency may change on resume; evaluation settings and selected data may not.
    mutable = {"concurrency", "timestamp", "last_started_at"}
    for key, value in settings.items():
        if key not in mutable and previous.get(key) != value:
            raise ValueError(
                f"Cannot resume: setting {key!r} differs. Use a different --out file."
            )
    if not isinstance(previous.get("timestamp"), str):
        raise ValueError("Existing output has no valid timestamp")
    settings["timestamp"] = previous["timestamp"]

    stored = document.get("items")
    if not isinstance(stored, list):
        raise ValueError("Existing output has no valid items list")
    selected = {item_key(item): item for item in items}
    records = {}
    for record in stored:
        if not isinstance(record, dict):
            raise ValueError("Existing output contains an invalid item")
        suite, identifier = record.get("suite"), record.get("id")
        if not isinstance(suite, str) or not isinstance(identifier, str):
            raise ValueError("Existing output contains an invalid item ID")
        key = suite, identifier
        if key not in selected or key in records:
            raise ValueError(f"Existing output contains an unknown or duplicate item: {key}")
        if type(record.get("correct")) is not bool:
            raise ValueError(f"Existing item {key} has invalid correctness")
        if not isinstance(record.get("extracted_answer"), str):
            raise ValueError(f"Existing item {key} has no valid extracted answer")
        if "completion_tokens" not in record or "finish_reason" not in record:
            raise ValueError(f"Existing item {key} is incomplete")
        tokens = record["completion_tokens"]
        if tokens is not None and (type(tokens) is not int or tokens < 0):
            raise ValueError(f"Existing item {key} has invalid token usage")
        seconds = record.get("seconds")
        if (
            type(seconds) not in (int, float)
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            raise ValueError(f"Existing item {key} has invalid seconds")
        if record["finish_reason"] is not None and not isinstance(
            record["finish_reason"], str
        ):
            raise ValueError(f"Existing item {key} has invalid finish_reason")
        if "error" in record and not isinstance(record["error"], str):
            raise ValueError(f"Existing item {key} has an invalid error")
        if record.get("category") != selected[key].category:
            raise ValueError(f"Existing item {key} has a mismatched category")
        records[key] = record
    return records, settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--budget-field", default=None)
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--suites", default=",".join(SUITES))
    parser.add_argument("--limit-per-suite", type=int, default=None)
    args = parser.parse_args()

    if args.concurrency < 1 or args.max_tokens < 1:
        parser.error("--concurrency and --max-tokens must be positive")
    if args.budget < 0:
        parser.error("--budget must be nonnegative")
    if args.limit_per_suite is not None and args.limit_per_suite < 1:
        parser.error("--limit-per-suite must be positive")
    if not args.model.strip():
        parser.error("--model cannot be empty")

    requested = [name.strip() for name in args.suites.split(",")]
    if not requested or any(name not in SUITES for name in requested):
        parser.error("--suites must be a comma-separated subset of " + ",".join(SUITES))
    if len(requested) != len(set(requested)):
        parser.error("--suites contains duplicates")
    args.suites = [suite for suite in SUITES if suite in requested]

    args.url = args.url.rstrip("/")
    parsed = urllib.parse.urlsplit(args.url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
    ):
        parser.error("--url must be an HTTP(S) base URL without a query or fragment")
    reserved = {
        "model", "messages", "temperature", "top_p", "top_k", "seed", "max_tokens"
    }
    if args.budget_field is not None:
        if not args.budget_field.strip() or args.budget_field in reserved:
            parser.error("--budget-field must be nonempty and cannot override request fields")

    args.data = args.data.resolve()
    args.out = args.out.resolve()
    if not args.data.is_dir():
        parser.error("--data must be an existing directory")
    return args


def main() -> int:
    args = parse_args()
    python3 = check_humaneval_support() if "humaneval" in args.suites else None
    items = load_items(args.data, args.suites, args.limit_per_suite)

    # Hash prompts, gold answers, tests, IDs, and order to reject changed data on resume.
    selection_json = json.dumps(
        [dataclasses.asdict(item) for item in items],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    now = utc_now()
    settings = {
        "url": args.url,
        "model": args.model,
        "budget_field": args.budget_field,
        "budget": args.budget,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "timestamp": now,
        "last_started_at": now,
        "data": str(args.data),
        "suites": args.suites,
        "limit_per_suite": args.limit_per_suite,
        "sampling_seed": SEED,
        "request_seed": 7,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "request_timeout_seconds": REQUEST_TIMEOUT,
        "connection_retries": 2,
        "seconds_measurement": "HTTP request including retries; excludes local grading",
        "selection_sha256": hashlib.sha256(selection_json).hexdigest(),
    }
    records, settings = load_existing(args.out, settings, items)

    def checkpoint() -> dict:
        summary = make_summary(items, records, args.suites)
        document = {
            "schema_version": SCHEMA_VERSION,
            "settings": settings,
            # Completion order varies with concurrency; file order does not.
            "items": [
                records[item_key(item)] for item in items if item_key(item) in records
            ],
            "summary": summary,
        }
        atomic_write(args.out, document)
        return summary

    pending = [item for item in items if item_key(item) not in records]
    checkpoint()
    print(
        f"{len(items)} selected; {len(records)} recorded; {len(pending)} remaining",
        file=sys.stderr,
    )

    interrupted = False
    new_count = 0
    futures = {}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency)
    try:
        for item in pending:
            futures[executor.submit(evaluate, item, args, python3)] = item
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            records[item_key(item)] = future.result()
            new_count += 1
            if new_count % 10 == 0:
                checkpoint()
                print(f"{len(records)}/{len(items)} recorded", file=sys.stderr)
    except KeyboardInterrupt:
        interrupted = True
        print(
            "Interrupted; cancelling queued work and waiting for active requests "
            "before saving. Active requests can take up to their timeout.",
            file=sys.stderr,
        )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        # Preserve results that finished while shutdown was waiting.
        for future, item in futures.items():
            key = item_key(item)
            if key not in records and future.done() and not future.cancelled():
                records[key] = future.result()
        summary = checkpoint()

    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 130 if interrupted else 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--_humaneval-helper":
        raise SystemExit(humaneval_helper(sys.argv[2]))
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"eval_suite.py: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)

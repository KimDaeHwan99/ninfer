#!/usr/bin/env python3
"""Same-prompt comparison of OpenAI-compatible local engines: speed, verifiable answers, tool calls.

    python3 bench-engines.py --name strata --url http://127.0.0.1:8080 --model qwen3.8-flash-next-iq3_s \
        --key-file /path/with/api_key.json --out results-strata.json

Every quality task is generated here with its answer computed by Python, so scoring needs no judge.
"""
import argparse
import datetime
import json
import random
import re
import time
import urllib.request
from pathlib import Path


def tasks():
    rnd = random.Random(20261022)
    out = []
    for i in range(10):                                      # multi-step arithmetic word problems
        a, b, c, d = rnd.randint(12, 90), rnd.randint(3, 9), rnd.randint(5, 40), rnd.randint(2, 6)
        out.append((f"가게에 사과 상자가 {a}개 있고 상자마다 사과가 {b}개 들어 있다. 그중 {c}개의 사과가 상했고, "
                    f"남은 사과를 {d}명이 똑같이 나누면 한 사람이 최대 몇 개를 받는가?", str((a * b - c) // d)))
    for i in range(8):                                       # date arithmetic
        start = datetime.date(2026, 1, 1) + datetime.timedelta(days=rnd.randint(0, 300))
        n = rnd.randint(20, 400)
        end = start + datetime.timedelta(days=n)
        out.append((f"{start.isoformat()}에서 {n}일 뒤의 날짜를 YYYY-MM-DD 형식으로 답하라.", end.isoformat()))
    for i in range(8):                                       # sorting
        nums = rnd.sample(range(100, 999), 7)
        out.append((f"다음 수를 작은 것부터 쉼표로 구분해 정렬하라: {', '.join(map(str, nums))}",
                    ",".join(map(str, sorted(nums)))))
    for i in range(7):                                       # counting
        word = "".join(rnd.choice("abcab") for _ in range(rnd.randint(18, 30)))
        out.append((f"문자열 '{word}'에서 글자 a는 몇 번 나오는가?", str(word.count("a"))))
    for i in range(7):                                       # number base / modular
        x, m = rnd.randint(1000, 99999), rnd.randint(7, 97)
        out.append((f"{x}를 {m}로 나눈 나머지는?", str(x % m)))
    return out


INSTR = "\n\n마지막 줄에 'ANSWER: <답>' 형식으로 답만 적어라."
TOOL = {"type": "function", "function": {"name": "set_alarm", "description": "Set an alarm.",
        "parameters": {"type": "object", "properties": {"hour": {"type": "integer"}, "minute": {"type": "integer"},
                                                         "label": {"type": "string"}},
                       "required": ["hour", "minute", "label"]}}}


def call(url, key, body, stream=False, timeout=1800):
    request = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(dict(body, stream=stream)).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    return urllib.request.urlopen(request, timeout=timeout)


def timed(url, key, model, prompt, max_tokens):
    """(prompt tokens, completion tokens, seconds to first token, decode seconds) from a streamed answer."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream_options": {"include_usage": True}}
    started, first, usage = time.monotonic(), None, {}
    with call(url, key, body, stream=True) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
            if first is None and (delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")):
                first = time.monotonic()
    end = time.monotonic()
    first = first or end
    return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), first - started, end - first


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--key-file", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--quality-tokens", type=int, default=4096)
    p.add_argument("--skip-quality", action="store_true", help="speed (and image) only, for screening settings")
    p.add_argument("--warm", type=int, default=0, help="generate this many tokens first (the adaptive tiers settle)")
    p.add_argument("--image", action="store_true", help="also time one small image request")
    a = p.parse_args()
    key = json.loads(Path(a.key_file).read_text()).get("api_key", "") if a.key_file else "none"
    res = {"engine": a.name, "model": a.model, "speed": {}, "quality": {}, "tools": {}}
    if a.warm:
        for n, prompt in enumerate(["게임 QA 자동화 도구의 설계 원칙을 자세히 설명하라.",
                                    "Write a Python module that parses game logs and reports frame stalls."]):
            timed(a.url, key, a.model, f"[warm {n} {time.time_ns()}]\n" + prompt, a.warm // 2)
    if a.image:
        import base64, struct, zlib
        def png(w=64, h=64):
            raw = b"".join(b"\x00" + b"\xff\x00\x00" * w for _ in range(h))
            chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
            return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                    + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
        url = "data:image/png;base64," + base64.b64encode(png()).decode()
        times = []
        for n in range(2):
            body = {"model": a.model, "max_tokens": 8, "temperature": 0, "messages": [{"role": "user", "content": [
                {"type": "text", "text": f"[{n} {time.time_ns()}] What color is this square? One word."},
                {"type": "image_url", "image_url": {"url": url}}]}]}
            started = time.monotonic()
            with call(a.url, key, body) as response:
                answer = json.loads(response.read())["choices"][0]["message"].get("content") or ""
            times.append(round(time.monotonic() - started, 2))
        res["image"] = {"seconds": times, "answer": answer.strip()[:20]}
        print("image", res["image"], flush=True)
    para = ("게임 QA 자동화에서는 실행 기록, 화면 증거, 입력 로그를 함께 남겨야 결함을 재현할 수 있다. "
            "Quality assurance logs must keep the build id, the case, and every observed frame. ") * 230
    for label, prompt, mt in [("prefill_10k", para + "\n위 글을 한 문장으로 요약하라.", 1),
                              ("decode_code", "파이썬으로 이진 탐색 트리를 구현하라. 삽입, 삭제, 중위 순회를 포함하고 주석을 달아라.", 512),
                              ("decode_korean", "게임 QA 자동화의 장단점을 한국어로 다섯 문단에 걸쳐 설명하라.", 512)]:
        # A fresh first line each run: no engine may answer from its prompt cache.
        runs = [timed(a.url, key, a.model, f"[측정 {label} {n} {time.time_ns()}]\n" + prompt, mt) for n in range(2)]
        pt, ct, ttft, dec = runs[-1]
        res["speed"][label] = {"prompt_tokens": pt, "completion_tokens": ct, "ttft_s": round(ttft, 2),
                               "prefill_tps": round(pt / ttft, 1) if ttft else None,
                               "decode_tps": round(ct / dec, 1) if dec and ct > 1 else None}
        print(label, res["speed"][label], flush=True)
    if a.skip_quality:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
        return
    right, details = 0, []
    started = time.monotonic()
    for question, answer in tasks():
        body = {"model": a.model, "messages": [{"role": "user", "content": question + INSTR}],
                "max_tokens": a.quality_tokens, "temperature": 0}
        try:
            with call(a.url, key, body) as response:
                text = json.loads(response.read())["choices"][0]["message"].get("content") or ""
        except Exception as error:
            text = f"ERROR {type(error).__name__}"
        got = re.findall(r"ANSWER:\s*(.+)", text)
        got = got[-1].strip().replace(" ", "") if got else ""
        ok = got == answer
        right += ok
        details.append({"q": question[:60], "want": answer, "got": got[:40], "ok": ok})
    res["quality"] = {"right": right, "total": len(details), "seconds": round(time.monotonic() - started),
                      "wrong": [d for d in details if not d["ok"]]}
    print("quality", right, "/", len(details), flush=True)
    good = 0
    for h in range(5):
        body = {"model": a.model, "messages": [{"role": "user", "content": f"{7 + h}시 {5 * h}분에 '회의 {h}' 알람을 맞춰 줘."}],
                "tools": [TOOL], "tool_choice": "auto", "max_tokens": 2048, "temperature": 0}
        try:
            with call(a.url, key, body) as response:
                msg = json.loads(response.read())["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            args = json.loads(calls[0]["function"]["arguments"]) if calls else {}
            good += args.get("hour") == 7 + h and args.get("minute") == 5 * h
        except Exception:
            pass
    res["tools"] = {"right": good, "total": 5}
    print("tools", good, "/ 5", flush=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

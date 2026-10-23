#!/usr/bin/env python3
"""Summarise a Basalt vs Ninfer record directory: speed (median of 3 runs), quality, public evals, telemetry per
phase. Writes summary.json and SVG charts (light/dark aware, hover titles) into OUT_DIR.

    python report.py RECORD_DIR OUT_DIR
"""
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

rec, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
ENGINES = ["basalt", "ninfer"]
LABEL = {"basalt": "Basalt", "ninfer": "Ninfer"}
SUITES = [("gsm8k", "GSM8K"), ("mmlupro", "MMLU-Pro"), ("humaneval", "HumanEval"), ("kmmlu", "KMMLU")]


def speed(engine):
    runs = [json.loads((rec / f"{engine}-speed{n}.json").read_text(encoding="utf-8")) for n in (1, 2, 3)]
    pick = lambda key, field: statistics.median(r["speed"][key][field] for r in runs)
    return {"prefill_tps": pick("prefill_10k", "prefill_tps"), "decode_code_tps": pick("decode_code", "decode_tps"),
            "decode_korean_tps": pick("decode_korean", "decode_tps"),
            "ttft_10k_s": pick("prefill_10k", "ttft_s"),
            "runs": [{k: r["speed"][k] for k in ("prefill_10k", "decode_code", "decode_korean")} for r in runs],
            "quality": runs[0]["quality"], "tools": runs[0]["tools"]}


def evals(engine):
    doc = json.loads((rec / f"{engine}-eval.json").read_text(encoding="utf-8"))
    items = doc["items"]
    tokens = sum(i["completion_tokens"] or 0 for i in items)
    return {"summary": doc["summary"], "completion_tokens": tokens, "items": len(items)}


def telemetry():
    rows = defaultdict(lambda: defaultdict(list))         # phase -> t -> rows of both GPUs
    with (rec / "telemetry.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[row["phase"]][row["t"]].append(row)
    result = {}
    for phase, samples in rows.items():
        complete = [s for s in samples.values() if len(s) == 2]
        if not complete:
            continue
        total_w = [sum(float(r["power_w"]) for r in s) for s in complete]
        num = lambda s, key: [float(r[key]) for r in s if r[key] != ""]
        result[phase] = {
            "seconds": len(complete),
            "gpu_power_avg_w": statistics.mean(total_w), "gpu_power_max_w": max(total_w),
            "gpu_energy_wh": sum(total_w) / 3600,
            "gpu_temp_max_c": {g: max(float(r["temp_c"]) for s in complete for r in s if r["gpu"] == g) for g in "01"},
            "gpu_temp_avg_c": statistics.mean(float(r["temp_c"]) for s in complete for r in s),
            "vram_max_mib": {g: max(float(r["vram_mib"]) for s in complete for r in s if r["gpu"] == g) for g in "01"},
            "gpu_util_avg_pct": statistics.mean(float(r["util_pct"]) for s in complete for r in s),
            "sm_clock_avg_mhz": statistics.mean(float(r["sm_mhz"]) for s in complete for r in s),
            "cpu_temp_max_c": max(sum((num(s, "cpu_c") for s in complete), []), default=None),
            "dimm_temp_max_c": max(sum((num(s, "dimm_c_max") for s in complete), []), default=None),
            "ram_used_max_gib": max(sum((num(s, "ram_used_gib") for s in complete), []), default=None),
        }
    return result


tel = telemetry()
data = {"telemetry": tel}
for e in ENGINES:
    d = {"speed": speed(e), "eval": evals(e)}
    ev = tel.get(f"{e}-eval")
    if ev:
        d["eval_tokens_per_wh"] = d["eval"]["completion_tokens"] / ev["gpu_energy_wh"]
        d["eval_minutes"] = ev["seconds"] / 60
    data[e] = d
(out / "summary.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")

# ---------------------------------------------------------------- charts
STYLE = """<style>
.bg{fill:#fcfcfb}.t{fill:#0b0b0b}.m{fill:#52514e}.g{stroke:#e4e3df}.ax{stroke:#a9a8a2}
.s0{fill:#2a78d6}.s1{fill:#eb6834}
@media (prefers-color-scheme: dark){.bg{fill:#1a1a19}.t{fill:#f0efec}.m{fill:#c3c2b7}.g{stroke:#2e2e2c}
.ax{stroke:#5f5e59}.s0{fill:#3987e5}.s1{fill:#d95926}}
text{font-family:system-ui,-apple-system,"Segoe UI","Malgun Gothic",sans-serif}
</style>"""


def nice_max(v):
    """Smallest round step (1, 2, 2.5, 5 x 10^k) whose four gridlines cover v."""
    step = min(s * 10 ** k for s in (1, 2, 2.5, 5) for k in range(-1, 6) if s * 10 ** k * 4 >= v)
    return step * 4, step


def chart(name, title, panels, unit_fmt="{:.0f}"):
    """panels: [(panel title, unit, [(group label, [basalt value, ninfer value])])] - one y scale per panel."""
    pw, ph, top, left = 300, 230, 64, 48
    width = left + len(panels) * (pw + 24)
    height = top + ph + 56
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
           f'height="{height}" role="img" aria-label="{title}">', STYLE,
           f'<rect class="bg" width="{width}" height="{height}" rx="8"/>',
           f'<text class="t" x="16" y="26" font-size="16" font-weight="600">{title}</text>']
    for i, e in enumerate(ENGINES):                                          # legend
        x = width - 190 + i * 90
        svg += [f'<rect class="s{i}" x="{x}" y="15" width="12" height="12" rx="3"/>',
                f'<text class="t" x="{x + 18}" y="26" font-size="13">{LABEL[e]}</text>']
    for p, (ptitle, unit, groups) in enumerate(panels):
        x0 = left + p * (pw + 24)
        vmax, step = nice_max(max(v for _, vals in groups for v in vals if v is not None) * 1.12)
        y = lambda v: top + ph - v / vmax * ph
        svg.append(f'<text class="t" x="{x0}" y="{top - 18}" font-size="13" font-weight="600">{ptitle}'
                   f'<tspan class="m" font-weight="400"> ({unit})</tspan></text>')
        k = 0
        while k * step <= vmax + 1e-9:
            gy = y(k * step)
            svg += [f'<line class="g" x1="{x0}" x2="{x0 + pw}" y1="{gy:.1f}" y2="{gy:.1f}"/>',
                    f'<text class="m" x="{x0 - 6}" y="{gy + 4:.1f}" font-size="11" text-anchor="end">'
                    f'{k * step:g}</text>']
            k += 1
        gw = pw / len(groups)
        bw = min(44, gw * 0.32)
        for g, (glabel, vals) in enumerate(groups):
            cx = x0 + gw * g + gw / 2
            for i, v in enumerate(vals):
                if v is None:
                    continue
                bx = cx - bw - 1 + i * (bw + 2)
                by, bh = y(v), top + ph - y(v)
                r = min(4, bh)
                svg += [f'<path class="s{i}" d="M{bx},{top + ph} V{by + r} Q{bx},{by} {bx + r},{by} '
                        f'H{bx + bw - r} Q{bx + bw},{by} {bx + bw},{by + r} V{top + ph} Z">'
                        f'<title>{LABEL[ENGINES[i]]} · {glabel}: {unit_fmt.format(v)} {unit}</title></path>',
                        f'<text class="t" x="{bx + bw / 2:.1f}" y="{by - 5:.1f}" font-size="11" '
                        f'text-anchor="middle">{unit_fmt.format(v)}</text>']
            svg.append(f'<text class="m" x="{cx:.1f}" y="{top + ph + 18}" font-size="12" '
                       f'text-anchor="middle">{glabel}</text>')
        svg.append(f'<line class="ax" x1="{x0}" x2="{x0 + pw}" y1="{top + ph}" y2="{top + ph}"/>')
    svg.append("</svg>")
    (out / f"{name}.svg").write_text("\n".join(svg), encoding="utf-8")


B, N = data["basalt"], data["ninfer"]
both = lambda f: [f(B), f(N)]
T = lambda e, phase: tel.get(f"{e}-{phase}", {})
chart("chart-speed", "속도 (높을수록 좋음, 3회 중앙값)", [
    ("프리필", "tok/s", [("10K 토큰 입력", both(lambda d: d["speed"]["prefill_tps"]))]),
    ("디코드", "tok/s", [("코드", both(lambda d: d["speed"]["decode_code_tps"])),
                         ("한국어", both(lambda d: d["speed"]["decode_korean_tps"]))])])
acc = lambda d, s: 100 * d["eval"]["summary"][s]["accuracy"]
chart("chart-intelligence", "지능 (정답률 %, 높을수록 좋음)", [
    ("공개 벤치마크", "%", [(lbl, both(lambda d, s=s: acc(d, s))) for s, lbl in SUITES]),
    ("자체 점검", "%", [("40문항", both(lambda d: 100 * d["speed"]["quality"]["right"] / d["speed"]["quality"]["total"])),
                       ("도구 호출", both(lambda d: 100 * d["speed"]["tools"]["right"] / d["speed"]["tools"]["total"]))])])
chart("chart-power", "전력과 효율 (GPU 2장 합계)", [
    ("평균 전력", "W", [(lbl, [T(e, ph).get("gpu_power_avg_w") for e in ENGINES])
                       for ph, lbl in (("idle", "대기"), ("speed", "속도 측정"), ("eval", "평가 4동시"))]),
    ("전력 효율", "tok/Wh", [("평가 생성 토큰", both(lambda d: d.get("eval_tokens_per_wh")))])])
chart("chart-thermal-memory", "발열과 메모리 (측정 구간 최대값)", [
    ("온도", "°C", [("GPU0", [T(e, "eval").get("gpu_temp_max_c", {}).get("0") for e in ENGINES]),
                    ("GPU1", [T(e, "eval").get("gpu_temp_max_c", {}).get("1") for e in ENGINES]),
                    ("CPU", [T(e, "eval").get("cpu_temp_max_c") for e in ENGINES]),
                    ("DDR5", [T(e, "eval").get("dimm_temp_max_c") for e in ENGINES])]),
    ("메모리", "GiB", [("VRAM 합계", [sum(T(e, "eval").get("vram_max_mib", {}).values()) / 1024 for e in ENGINES]),
                      ("시스템 RAM", [T(e, "eval").get("ram_used_max_gib") for e in ENGINES])])], "{:.1f}")
print(json.dumps({e: {"speed": {k: v for k, v in data[e]["speed"].items() if k != "runs"},
                      "acc": {s: data[e]["eval"]["summary"][s]["accuracy"] for s, _ in SUITES}} for e in ENGINES},
                 ensure_ascii=False))

# Engine comparison scripts

The scripts behind [Basalt vs Ninfer on 2x RTX 5060 Ti](../../docs/basalt-vs-ninfer-2x5060ti.md). Each one talks to
an OpenAI-compatible `/v1/chat/completions` server, so they work for other local engines too.

| Script | What it does |
|---|---|
| `bench_engines.py` | Speed (10K-token prompt, 512-token code and Korean answers), 40 generated tasks with answers computed by the script, 5 tool calls |
| `eval_suite.py` | GSM8K (100), MMLU-Pro (10 per category, 140), HumanEval (every other task, 82, executed in a sandboxed subprocess) and KMMLU (100) with a fixed seed, so every engine gets the same items; resumable |
| `telemetry.py` | Once a second: per-GPU power, temperature, VRAM, utilisation and clocks, CPU (k10temp) and DDR5 (spd5118) temperatures, used RAM, tagged with the current phase |
| `report.py` | Medians of the three speed runs, eval accuracy, per-phase telemetry (average, maximum, energy, tokens per Wh) and the SVG charts |
| `run_compare.sh` | The whole run for both engines, one after the other |

Datasets go in `evalsets/` (or `EVALSETS=`), from Hugging Face:

| File | Source |
|---|---|
| `gsm8k-test.parquet` | `openai/gsm8k`, `main` test split |
| `mmlu-pro-test.parquet` | `TIGER-Lab/MMLU-Pro`, test split |
| `humaneval.parquet` | `openai/openai_humaneval`, test split |
| `kmmlu-<Subject>.csv` | `HAERAE-HUB/KMMLU`, test split of each subject (the published run used Accounting, Biology, Chemistry, Computer-Science, Economics, Electrical-Engineering, Law, Psychology) |

`eval_suite.py` needs `pyarrow`; point `PY=` at an interpreter that has it. The other scripts use only the standard
library. Set `BASALT_DIR`, `BASALT_CONFIG`, `NINFER_DIR` and `NINFER_ARTIFACT`, and the library path Ninfer needs on
your machine, before `sh run_compare.sh`.

The published run used the same steps and settings, with Basalt's speed measured on the already running production
service instead of the separate instance that `run_compare.sh` starts.

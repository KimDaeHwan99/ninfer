#!/bin/sh
# Basalt vs Ninfer on one machine: speed (3 runs), the 40-task check and tool calls, four public evals at four
# concurrent requests, and one-second telemetry for the whole run. Each engine runs alone on both GPUs.
# Set the paths below for your machine; PY must have pyarrow (eval_suite.py reads the parquet datasets).
#   sh run_compare.sh
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HERE/records/$(date +%F)-basalt-vs-ninfer}
EVALSETS=${EVALSETS:-$HERE/evalsets}
PY=${PY:-python3}
BASALT_DIR=${BASALT_DIR:-$HOME/basalt}
BASALT_CONFIG=${BASALT_CONFIG:-$BASALT_DIR/basalt-serve.json}       # your serving config (model, gpu, context)
BASALT_MODEL=${BASALT_MODEL:-qwen3.8-flash-next-iq3_s}                # the config's alias
NINFER_DIR=${NINFER_DIR:-$HOME/ninfer}
NINFER_ARTIFACT=${NINFER_ARTIFACT:-$NINFER_DIR/models/qwen3_8_27b_quasar_nvfp4.ninfer}
NINFER_MODEL=${NINFER_MODEL:-qwen3.8-27b-quasar}

mkdir -p "$OUT"
PH="$OUT/phase.txt"
phase() { echo "$1" > "$PH"; echo "PHASE $1 $(date +%H:%M:%S)"; }
waitup() { for i in $(seq 1 400); do curl -sf -m 5 "$1/health" >/dev/null && return 0; sleep 3; done; return 1; }
speed() {  # $1 engine $2 url $3 model
  python3 "$HERE/bench_engines.py" --name "$1-1" --url "$2" --model "$3" --out "$OUT/$1-speed1.json" --warm 1024
  for n in 2 3; do
    python3 "$HERE/bench_engines.py" --name "$1-$n" --url "$2" --model "$3" --out "$OUT/$1-speed$n.json" --warm 1024 --skip-quality
  done
}
evals() {  # $1 engine $2 url $3 model, then extra eval_suite options
  e=$1 url=$2 model=$3; shift 3
  "$PY" "$HERE/eval_suite.py" --url "$url" --model "$model" --data "$EVALSETS" --out "$OUT/$e-eval.json" \
    --concurrency 4 --max-tokens 6144 "$@"
}
python3 "$HERE/telemetry.py" "$OUT/telemetry.csv" "$PH" & TEL=$!

# ---- Basalt: speed with the serving config as is (one request at a time), evals with four slots
python3 - "$BASALT_CONFIG" "$OUT" <<'EOF'
import json, pathlib, sys
config = json.loads(pathlib.Path(sys.argv[1]).read_text())
out = pathlib.Path(sys.argv[2])
config["server"].update(port=8091); config["server"].pop("idle-unload", None)
(out / "basalt-speed-config.json").write_text(json.dumps(config, indent=1))
config["server"].update(parallel=4)
(out / "basalt-eval-config.json").write_text(json.dumps(config, indent=1))
EOF
for stage in speed eval; do
  phase "basalt-$stage-load"
  (cd "$BASALT_DIR" && bin/basalt serve --config "$OUT/basalt-$stage-config.json" > "$OUT/basalt-$stage-server.log" 2>&1) & B=$!
  waitup http://127.0.0.1:8091
  if [ "$stage" = speed ]; then
    curl -s -m 600 -X POST -H "Content-Type: application/json" -d "{}" http://127.0.0.1:8091/load >/dev/null
    phase basalt-idle; sleep 60
    phase basalt-speed; speed basalt http://127.0.0.1:8091 "$BASALT_MODEL"
  else
    phase basalt-eval; evals basalt http://127.0.0.1:8091 "$BASALT_MODEL" --budget-field reasoning_budget_tokens --budget 2048
  fi
  kill $B; wait $B 2>/dev/null; sleep 20
done

# ---- Ninfer: the recommended two-GPU command (thinking budget 2048 set on the server)
phase ninfer-load
"$NINFER_DIR/build/apps/ninfer-serve" "$NINFER_ARTIFACT" \
  --host 127.0.0.1 --port 8092 --tp 2 --devices 0,1 --kv-dtype fp8 --max-context 131072 --kv-capacity auto \
  --max-concurrency 4 --host-cache-mib 16384 --default-thinking-budget 2048 --prefill-chunk 8192 --spec mtp \
  --draft-tokens 4 --lm-head-draft --ngram-draft-tokens 15 --ngram-min-match 12 --vision --vram-headroom-mib 512 \
  --pending-timeout-ms 600000 > "$OUT/ninfer-server.log" 2>&1 & N=$!
waitup http://127.0.0.1:8092
phase ninfer-idle; sleep 60
phase ninfer-speed; speed ninfer http://127.0.0.1:8092 "$NINFER_MODEL"
phase ninfer-eval; evals ninfer http://127.0.0.1:8092 "$NINFER_MODEL"
kill $N; wait $N 2>/dev/null
phase done; sleep 5; kill $TEL; rm -f "$PH"
python3 "$HERE/report.py" "$OUT" "$OUT/report"

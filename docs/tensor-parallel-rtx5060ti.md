# Two-GPU tensor parallelism on RTX 5060 Ti

This fork's `tp2-5060ti` branch adds `--tp 2` to NInfer. It splits the Qwen3.8-27B NVFP4 package
across two 16 GB GeForce RTX 5060 Ti cards, so the model fits and runs on cards that cannot share
memory peer-to-peer.

The branch was written by **Claude Opus 5.5 (medium reasoning effort)** in Claude Code. A human
operator set the goals, approved each production change and ran the server. Every commit carries a
`Co-Authored-By: Claude Opus 5.5` trailer.

> 한국어 요약 (2026-10-03 최종): RTX 5060 Ti 16GB 두 장(P2P 없음)에서 Qwen3.8-27B를 텐서 병렬로
> 서빙하는 브랜치입니다. 모든 코드는 Claude Code의 Claude Opus 5.5가 작성했습니다. 권장 구성은
> QUASAR-QAT 전체 NVFP4 아티팩트 + MTP(드래프트 4) + n-gram 복사 드래프트 + hybrid prefix 캐시입니다.
> 최종 측정(라우터 경유, 공식 아티팩트 구성 대비): 코드 생성 94.9 → 117~120 tok/s, 한국어 67.6 → 76~78,
> 1.6만/3.1만 토큰 첫 응답 2.70/5.71 → 2.38/5.20초, 짧은 요청 첫 응답 0.10 → 0.05~0.10초, 파일을 거의
> 그대로 다시 쓰는 편집 128 → 379 tok/s, 대화 재개 첫 응답 0.125 → 0.055초, 동시 4요청 합계
> 264 → 300 tok/s, KV 용량 114K → 223K 토큰. 품질은 GSM8K 100문항 98, MMLU-Pro 210문항 174(공식
> 아티팩트 172)로 같은 수준입니다. 아래 "2026-10-03 update"에 측정과 근거가 있고, 그 아래 절들은
> 공식 아티팩트 시절의 기록입니다.
>
> 그 뒤 `--tp 2`에서는 토큰 임베딩을 VRAM 대신 pinned 시스템 RAM에 두는 것이 기본값이 되었습니다
> (`--embedding-offload`). 카드당 1.185 GiB를 비워 KV 풀이 약 7만 토큰 늘어날 것으로 계산되지만 아직
> 실측 전입니다. 위 측정은 임베딩이 VRAM에 있던 상태이며 `--embedding-offload off`로 재현합니다.
> 또 TP 링크의 장치 버퍼(카드당 240 → 180 MiB)가 이제 KV 계획 안에 잡혀, 더 이상 1 GiB 헤드룸을
> 차지하지 않습니다. 같은 헤드룸에서 KV 풀은 그만큼(약 1.1만 토큰) 작아집니다(미측정).

## 2026-10-03 update

The branch now also carries
[Wallawalla47/ninfer-custom](https://github.com/Wallawalla47/ninfer-custom) (89 commits on the same
upstream base: hybrid prefix cache, ngram copy drafting, programmatic-dependent decode launches,
tool-call and serving fixes, CUDA Graph allowance from measurement) and two-GPU work that makes those
features and more artifacts run at `--tp 2`. Ideas and one technique (compiling the parent kernels a
second time for a shard) come from [ValerioDolci/ninfer-tp2](https://github.com/ValerioDolci/ninfer-tp2).

Recommended serving configuration (production since 2026-10-03):

```bash
hf download Feyd89/Qwen3.8-27B-QUASAR-QAT-nvfp4-NInfer qwen3_8_27b_quasar_nvfp4.ninfer --local-dir models
ninfer-serve models/qwen3_8_27b_quasar_nvfp4.ninfer --host 0.0.0.0 --port 8080 \
  --tp 2 --devices 0,1 --kv-dtype fp8 --max-context 131072 --kv-capacity auto \
  --max-concurrency 4 --host-cache-mib 16384 --default-thinking-budget 2048 \
  --prefill-chunk 8192 --spec mtp --draft-tokens 4 --lm-head-draft \
  --ngram-draft-tokens 15 --ngram-min-match 12 --vision
```

This command was measured with the token embedding in VRAM; the build now keeps it in system RAM
at `--tp 2` (see [Token embedding in system RAM](#token-embedding-in-system-ram)), and
`--embedding-offload off` restores the measured placement. Agent clients that send bursts of
requests may also want `--pending-timeout-ms 600000`: with the default of 30 s, a request that has
not started within 30 s of arriving (while waiting for one of the four lanes, for example) is
answered with HTTP 503 `request_queue_timeout`, or HTTP 504 `timeout_error` on the Anthropic
Messages endpoint. That flag is not part of the measured configuration.

What changed, each measured on the machine below against the previous production
(official artifact, original prefix cache, image `dbd162d1`):

| Change | Effect on 2x RTX 5060 Ti |
|---|---|
| NVFP4 shards of the attention/GDN input and [5120,3072] output projections | All-NVFP4 artifacts run at `--tp 2`. QUASAR-QAT weights take 8.85 GiB per card instead of 10.8: code 94.9 → 106 tok/s, Korean 67.6 → 73, 16K/31K-token first token 2.70/5.71 → 2.43/5.22 s, KV pool 114K → 248K tokens. Shards tested against the FP64 oracle (attention) and bit for bit against the parent (GDN). |
| Hybrid prefix cache at `--tp 2` | Both ranks publish Host transfers up to the frontier complete on both (deterministic poll), so their prefix indexes stay identical. Three conversations alternating over 12K-token documents: resumed-turn first token 0.107 → 0.041 s. |
| ngram copy drafting at `--tp 2` + two-width MTP graph families | Copy rounds verify 16 wide, every other round 4 wide on the same frame. Rewriting a source file: 128 → 396 tok/s with ordinary generation unchanged (before the two-width families, ngram cost ordinary generation 10-13 %). |
| Admission search budget back to upstream's | Wallawalla's 250 ms search spent 40-128 ms per short request once long conversations were cached and found nothing: short-request first token 0.19-0.25 → 0.06 s. |
| `ninfer-serve` exits with status 2 after an Engine-wide failure | A container restart policy now reloads a dead engine. |
| `--draft-tokens 4` instead of 3 (with QUASAR and the two-width families) | Three interleaved runs per setting: greedy code 102-106 → 117-120 tok/s, Korean 74.5-75.5 → 73.2-75.9, sampled essay mean 91.1 → 89.8 (−1.4 %, within the run-to-run spread of 84-96), copy edits 388 → 379-386, GSM8K 100 at four parallel requests 96 → 97 correct in 115 → 112 s. Five drafts were no faster than four on code and slower on Korean and copy edits. The older `--draft-tokens 3` advice below predates QUASAR and the two-width families. |
| DFlash2 at `--tp 2` (replicated drafter) | Works (official artifact: code +31 %, copy edits +65 % over MTP3), but not recommended here: with QUASAR the grafted drafter accepts fewer drafts (Korean 12 % vs 20 %) and costs 60 % of the KV pool. |

Final measurement of the recommended configuration (2026-10-03, through the router, three runs):
greedy code 117.1-119.6 tok/s, Korean 76.3-77.5, sampled essay mean 89.4, 16K/31K first token
2.38/5.20 s, copy edits 379 tok/s, resumed conversation 0.055 s, parallel sampled requests
1/2/4 -> 107.5/165.1/300.2 tok/s aggregate, GSM8K 100 98, MMLU-Pro 210 174.

Quality of the recommended configuration: GSM8K 200 (zero-shot, greedy) 193-196 against 195 for the
previous production; MMLU-Pro 210 (15 per category, thinking budget 2048) 175 against 172 for the
official artifact on the same build. Not adopted, with measurements: `nvfp4` KV, earlier A4 routes,
ValerioDolci's pipelined mailbox kernel and wide mailbox slots (our slot is sized for the widest
exchange, so no staged copies remain), ValerioDolci's four-rows-per-warp [5120,8704] GEMV for
T=3..5 (on the RTX 5060 Ti 65.5 → 67.6 µs at T=3/4 and 67.6 → 71.7 µs at T=5, slower than the
sliced-K route kept here), shorter TP timeouts, and a second Wallawalla47 merge (39 commits; equal
speed in an interleaved A/B, kept on branch `tp2-5060ti-walla2`). Benchmarks: `bench/gsm8k_eval.py`, `bench/multiturn.py`, `bench/copy_edit.py`
in the operator's workspace; the commits list their own measurements.

## Token embedding in system RAM

`--tp 2` keeps the token embedding table in pinned system RAM by default (`--embedding-offload
auto`, which is `on` at `--tp 2` and `off` on one GPU). Before, each rank held its own replicated
copy in VRAM. The QUASAR artifact stores the table as FP8 rows: 248,320 × 5,120 bytes plus one BF16
scale per row, 1.185 GiB per card. The embedding gather reads only the rows of the current tokens
through the pinned block's mapped address, so the table never crosses PCIe as a whole.

Expected effect, from arithmetic only; **not yet measured on the machine below**:

| Item | Expected |
|---|---|
| KV pool (`--kv-capacity auto`) | about +70K tokens: 1.185 GiB at about 17.5 KB of FP8 KV per token per rank (16 attention layers plus the MTP layer, 2 KV heads each) |
| Decode | each gather reads the current rows (5 KiB per token) over PCIe: a few microseconds per round of about 25-30 ms, under 0.1 % |
| Prefill | an 8,192-token chunk gathers 42 MB twice with MTP (text and MTP inputs), a few milliseconds per chunk, well under 1 % of the first token |
| System RAM | +1.185 GiB pinned per rank (2.37 GiB in total, since each rank pins its own copy); the loader reads the table straight into the pinned block |
| Output | unchanged: the same kernel reads the same bytes |

Measure before relying on the larger pool: the KV capacity printed at startup, greedy code and
Korean tok/s, the 16K and 31K first token, and `ninfer-perplexity --quick` at `--tp 2`.
`ninfer-perplexity` takes the same flag, so the PPL table above can be checked with either placement.
`--embedding-offload off` restores the VRAM table. `--embedding-offload on` also works on one GPU,
but not together with `--vision-offload on`, whose eviction ladder borrows the device-resident
table. An artifact whose output head shares the embedding's stored object keeps the table in VRAM
for the head, so offload leaves it there. Startup fails with a message if the pinned block is not
mapped at the same address on the device (UVA), which the tensor-parallel link already requires.

## Link buffers in the KV plan

The two-GPU link keeps device buffers on each card: a receive scratch for the copy-channel
all-reduce and two partial buffers of one slot each. A slot holds the widest exchange, here one
8,192-column prefill chunk of BF16 hidden states (80 MiB). Before, the link was created after
`--kv-capacity auto` had measured free memory, so its 240 MiB per card came out of the 1 GiB
headroom (about 784 MiB was really left), and a small `--vram-headroom-mib` could fail startup.
Now the link is created before that measurement, and the receive scratch is one piece wide (20 MiB)
instead of a whole slot, since the receive stream combines each piece before the next one lands:
180 MiB per card, inside the KV plan.

Expected, not yet measured: the link no longer takes 240 MiB of the 1 GiB headroom, so lowering
`--vram-headroom-mib` becomes safe to try (kernel modules CUDA loads on first use still come out of
the headroom; see the next section), and at the same headroom the auto KV pool is about 11K tokens
smaller than before (180 MiB at about 17.5 KB per token). At `--tp 2` the startup figures labelled
"after weights" (the debug memory ledger and KV sizing errors) now include the link's 180 MiB.

## Tuning `--vram-headroom-mib`

`--kv-capacity auto` leaves 1 GiB per card unused by default. How much of it the server really
needs depends on the build and the workload: CUDA loads kernel modules on first use, and Vision,
long prompts and wide n-gram rounds touch more of them. Every 100 MiB moved from the headroom into
the pool is about 6K KV tokens (17.5 KB per token per rank). Measure on the serving machine:

1. Start the server with the production command and the default headroom.
2. From the repository, run the measurement against it:

   ```bash
   python3 tools/bench/measure_vram_headroom.py --base-url http://127.0.0.1:8080 \
     --gpus 0,1 --headroom-mib 1024 --output profiles/bench/headroom
   ```

   It samples nvidia-smi on both cards every 100 ms while it sends four concurrent copy edits
   (wide n-gram verify rounds), a 64K-token prompt, four heterogeneous concurrent requests, a
   28-image request and two heavy media requests, then the copy edits again. Without `--vision`,
   pass `--no-media`, which also drops the heterogeneous set because it includes an image. It
   prints the minimum free memory per card and the headroom that keeps 256 MiB (`--safety-mib`) of
   it free, and writes `samples.csv`, `report.json` and the load cases under `cases/` (load only,
   not TTFT evidence). It exits 2 when a load step failed or ran short, and gives no value when
   more than the headroom stayed free, which means `--headroom-mib` or `--gpus` does not match the
   server. Set `NINFER_API_KEY` for a server started with `--api-key`.
3. Restart with `--vram-headroom-mib <recommended>`, run the tool again with
   `--headroom-mib <recommended>`, and keep the value only if the minimum stays above the safety
   margin. The startup line `capacity | KV ... | free ...` reports the smaller of the two cards'
   free memory at `--tp 2`.

Runtime buffers are allocated at startup, so a headroom that is too small shows up when a lazily
loaded kernel module or a launch needs memory that is not there: `ninfer-serve` prints a
`CUDA_CHECK(...) failed` line and aborts (SIGABRT), losing the in-flight requests and every cached
prefix, Device and Host. Repeat the measurement after every rebuild or option change. Peaks shorter
than the sample period are not seen; the safety margin covers them. To bound what lazy module
loading can still take, start twice at the same headroom, once normally and once with
`CUDA_MODULE_LOADING=EAGER` (every module loaded before the KV pool is sized): the drop in KV
tokens times about 17.5 KB per token is the most that lazy loading adds after startup.

## Results

Measured on one machine: Ryzen 5 9600X, DDR5-5600 2×32 GB, 2× RTX 5060 Ti 16 GB on PCIe 5.0 x8/x8,
driver 595.91.07, CUDA 13.1.2. The baseline is
[lynx-gt/ninfer-tp2-5060ti](https://github.com/lynx-gt/ninfer-tp2-5060ti) at `444558d` with the same
model and options (int8 KV, which was its production setting).

| Metric | Baseline | This branch |
|---|---:|---:|
| Sampled chat, 1500 tokens | 67.1 tok/s | ≈77 tok/s |
| MTP round (4-token verify) | 33.4 ms | 29.4 ms |
| Greedy code, 1500 tokens | 78.4 tok/s | 94.2 tok/s |
| First token, 15,821-token prompt | 5.21 s | 2.7 s |
| First token, 31,306-token prompt | – | 5.7 s |
| First token, 59,417-token prompt | – | 12.7 s |

Greedy tok/s depends on how many MTP drafts the text accepts, so the round time is the fairer
hardware comparison. Quick-corpus perplexity at `--tp 2` (`ninfer-perplexity --quick`):

| KV dtype | PPL |
|---|---:|
| bf16 | 4.3127 |
| fp8 | 4.3123 |
| int8 | 4.3140 |
| nvfp4 | 4.3188 |
| k8v4 | 4.3211 |

## Usage

Requirements are those of upstream NInfer, but with two `sm_120a` GPUs (tested on RTX 5060 Ti 16 GB)
instead of one RTX 5090. Build as usual.

Use the official v3 artifact as published. No conversion is needed, and the branch was verified with
it (SHA256 `74d2c571…bbb77d82`, same speed as the results below):

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer --local-dir Qwen3.8-27B-nvfp4-NInfer
```

An older v2 download also works after `python3 tools/upgrade_ninfer_v2_to_v3.py <v2> <v3>`. The
results below were first measured that way.

Serving configuration of the measurements in this and the following sections. It predates the
2026-10-03 command above, which supersedes it: it uses the original prefix cache, whose options
`--device-state-slots`, `--host-kv-mib` and `--host-state-slots` are rejected unless
`--use-original-prefix-caching` selects that cache, and three MTP drafts.

```bash
ninfer-serve Qwen3.8-27B-nvfp4-NInfer/qwen3_8_27b_nvfp4.ninfer --host 0.0.0.0 --port 8080 \
  --tp 2 --devices 0,1 \
  --kv-dtype fp8 --max-context 131072 --kv-capacity auto \
  --max-concurrency 4 --use-original-prefix-caching \
  --device-state-slots 2 --host-kv-mib 16384 --host-state-slots 16 \
  --default-thinking-budget 2048 \
  --prefill-chunk 8192 --spec mtp --draft-tokens 3 --lm-head-draft --embedding-offload off
```

| Option | Why |
|---|---|
| `--kv-dtype fp8` | Same perplexity as bf16 KV (table above), fastest long-context prefill. |
| `--max-context 131072` | The auto-sized KV pool holds 140,608 tokens, so one request may use up to 131,072 of them. Raising the ceiling from 65,536 did not shrink the pool. |
| `--kv-capacity auto` | Uses the VRAM left after weights and runtime: 140,608 KV tokens instead of 65,536, so more conversations and shared prompts stay cached on the GPU. Leaves 1 GiB headroom (14.8 GB used per card). |
| `--max-concurrency 4` | The 27B weights already sit entirely on the GPUs, so spare memory cannot speed up a single request; it buys parallel requests instead (table below). |
| `--host-kv-mib 16384 --host-state-slots 16` | Pinned host cache for conversations that leave the GPU, doubled from the 8 GiB / 8-slot default. The 64 GB host keeps 34 GB available with the server running. |
| `--default-thinking-budget 2048` | Caps thinking so the answer fits the request's `max_tokens`. At the cap the engine closes thinking and the model answers (table below). Per-request overrides use Anthropic `thinking.budget_tokens`. |
| `--prefill-chunk 8192` | Faster first token on long prompts than 4096 (31,306 tokens 6.2 → 6.0 s, 59,417 tokens 14.0 → 13.7 s, measured before the last attention change); short prompts are unaffected. Wider chunks are rejected at `--tp 2`: 16384 cost 20 % of the KV pool (223K → 177K tokens) and deadlocked the two ranks' copy-channel all-reduce on the first prompt longer than 8192 tokens. |
| `--draft-tokens 3 --lm-head-draft` | Best measured MTP setting for single requests with the official artifact (with QUASAR and the two-width graph families four drafts are faster; see the 2026-10-03 update); 2 and 4 drafts were slower on average, and turning off the reduced draft head was about 10% slower. With four parallel requests 2, 3 and 4 drafts all reached about 200 tok/s. |

`ninfer` (CLI) and `ninfer-perplexity` accept the same `--tp 2 --devices 0,1`.

### Concurrency

Sampled 800-token essays sent in parallel to the recommended configuration:

| Parallel requests | Aggregate output |
|---:|---:|
| 1 | 77.9 tok/s |
| 2 | 144.6 tok/s |
| 4 | 263.9 tok/s |

A single request runs as before (greedy code 95.0 tok/s, 31K-token prompt first token 5.8 s).
Batched decoding changes floating-point order, so a greedy answer generated alongside others can
diverge from the same request served alone after a few dozen to a few hundred characters. Accuracy on
the hard set below did not drop when four requests ran at once.

### Thinking budget

With `max_tokens` 4096, long thinking used the whole budget and the answer was cut off. Measured
with four parallel requests (temperature 0.6, top_p 0.95, top_k 20; one sample per item):

| `--default-thinking-budget` | MMLU-Pro (42) | HumanEval (20, executed) | Cut off |
|---|---:|---:|---:|
| none | 30 | 18 | 9 |
| 3072 | 33 | 19 | 0 |
| **2048** | **34** | **20** | **0** |
| 1024 | 31 | 20 | 0 |

On AIME 2025 problems 6-15 (`max_tokens` 14000), no budget scored 3/10 with 7 answers cut off and
119 s per problem. A 2048 budget scored 4/10 with none cut off and 51 s per problem. These are
small single-sample sets, so differences of one or two items are within sampling noise.

### Image input (`--vision`)

`--tp 2` accepts `--vision`. The Vision encoder is not split: both GPUs hold a full copy
(+0.3 GiB each) and encode the same media, so each rank's replicated hidden state stays identical
without a cross-GPU transfer. On 16 GB cards the copy and the Vision workspace leave room for
`--max-context 98304` and an auto KV pool of 114,176 tokens, compared with 131,072 and 140,608 without
Vision. With 131,072 the minimum runtime reservation no longer fits. These limits are for the official
artifact; the all-NVFP4 QUASAR artifact (8.85 GiB of weights per card with the embedding in VRAM,
about 7.67 GiB with the default `--embedding-offload`) runs `--vision` at `--max-context 131072`,
as in the 2026-10-03 command.

```bash
ninfer-serve ... --tp 2 --devices 0,1 --vision --max-context 98304 --kv-capacity auto ...
```

Checked with the bundled example images (scene description, chart text and shapes, a two-image
difference). All answers matched the images. Text-only speed was unchanged: greedy code 95.0 tok/s,
first token for a 15,840-token prompt 2.68 s. Repeating the same greedy image request is not
byte-identical, because the second run reuses cached media and prefix state.

## What the branch changes

1. **TP2 runtime.** Each GPU runs a full program over its weight shard (SPMD). Attention q/gate/k/v,
   GDN q/k/v/z/a/b, MLP gate/up and both output heads are column-split. The attention, GDN and MLP
   output projections are row-split and summed by a residual all-reduce. Norms and the MTP input
   projection are replicated in VRAM; the token embedding is replicated in pinned system RAM unless
   `--embedding-offload off` keeps it in VRAM.
2. **Collectives over host memory.** Without P2P, every byte crosses host DRAM. Small decode-size
   all-reduces use 8-byte lines that carry 4 payload bytes plus a sequence tag, so one PCIe round trip
   delivers both. They are CUDA Graph capturable (3.7 µs for one 5120-wide column).
3. **Prefill overlap.** Large all-reduces run on copy-engine side streams in up to four column
   slices. Each slice overlaps the next slice's FFN and the next layer's column-wise prologue. This
   cut stream idle time from 17% to 7.5% of a prefill.
4. **Kernel routes for shard shapes.** NVFP4 TMA routes and a fused SwiGLU for the
   17408×5120 / 5120×8704 MLP shards (gate/up 2× faster at prefill width). A 24-head GDN control
   MMA (5.6× faster). FP8 split-K waves sized for 36 SMs.
5. **Device-aware attention.** Causal KV split budgets assumed 170 SMs; they now read the current
   device. FP8 prefill attention accumulates PV per key tile in FP16. Consumer Blackwell runs
   FP32-accumulating MMAs at half rate, so 16K-key attention dropped from 7.01 to 5.22 ms.

## Hardware fit audit

The serving options started from the baseline project's production command. Each was rechecked
against this machine (36-SM GPUs, 16 GB each, 64 GB host) instead of being copied:

| Item | Before | Now | Evidence |
|---|---|---|---|
| KV pool | fixed 65,536 tokens | `auto`, 140,608 tokens | ~2.6 GB per GPU was unused |
| Request ceiling | 65,536 | 131,072 | pool unchanged |
| Concurrency | 1 | 4 | 3.4x aggregate output, single request unchanged |
| Host conversation cache | 8 GiB, 8 slots | 16 GiB (8 GiB pinned per rank), 16 slots | 34 GB host memory still free |
| Thinking | unlimited | 2048-token budget | no cut-off answers, better scores |
| FP8 attention/GDN input shards | split-K waves for 170 SMs | 36-SM waves through 1024 / 2048 tokens | 4-11% faster on those widths. Wider inputs keep the old plan, which measured as fast or faster. No change in 16K/31K first-token time. |

Checked and left as is:
- **RMSNorm prefetch threshold (170 blocks).** It only matters for 24-113-token gated norms, which this workload rarely hits.
- **Context-cost model.** It has no RTX 5060 Ti preset, so the generic coefficients apply. They overprice recompute by about 1.4x and underprice host transfer by about 2.6x for this TP2 machine. Restoring a cached 4K-token context (~10 ms) still beats recomputing it (~700 ms) by far more than that error, so no cache decision changes.
- **CPU governor, PCIe ASPM, GPU power limit.** No measurable effect.

## Findings that may help other consumer-GPU setups

- With both GPUs copying at once, each gets about 15.5 GB/s per direction. One GPU alone gets 28.7.
  Host memory bandwidth is the limit, whichever engine (SM loads or copy engines) moves the data.
- Measured `mma.sync` peaks on RTX 5060 Ti: FP16→FP32 54, FP16→FP16 109, E4M3→FP32 109,
  E4M3→FP16 218 TFLOPS.
- Decode reaches about 95% of attainable memory bandwidth.

## Comparison with Strata and Swift 1.5 Qwen3.8-Flash-Next

Same machine and prompts. [Strata](https://github.com/Niko1221/Strata) v0.1.32 serves
[Swift 1.5 Qwen3.8-Flash-Next GSQ-RCO](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
(125B MoE, 6B active). It splits layers across both GPUs, keeps experts in RAM and uses int8 KV and
MTP. It fills both GPUs (15.5 GB each) and 40-46 GB of RAM, so it cannot run beside ninfer. Strata
was tuned with its own calibration (`tools/calibrate.py`). Both servers used a 2048-token thinking
budget; ninfer used the recommended configuration above.

| Test | ninfer, 27B NVFP4 | Strata IQ2_XS (2.4 bpw) | Strata IQ3_XXS (3.1 bpw) |
|---|---:|---:|---:|
| Greedy code, 1500 tokens | 95.2 tok/s | 86.8 tok/s | 90.6 tok/s |
| Sampled essay | 79.0 tok/s | 82.4 tok/s | 78.6 tok/s |
| Korean explanation | 64.1 tok/s | 91.6 tok/s | 77.2 tok/s |
| First token, 15,840-token prompt | 2.65 s | 7.69 s | 7.58 s |
| First token, 31,325-token prompt | 5.71 s | 12.09 s | 11.73 s |
| AIME 2025 #6-15 | 4/10 | 3/10 | 3/10 |
| MMLU-Pro 42 | 34/42 | 33/42 | 35/42 |
| HumanEval 20 | 20/20 | 19/20 | 19/20 |
| Truncated answers (72 items) | 0 | 0 | 0 |
| Parallel requests | 4 | 1 | 1 |

Calibration chose PCIe share 0 and draft floor 0.7 for IQ2_XS, about 4% faster decode, and 3 CPU
workers for IQ3_XXS. Without calibration and without a thinking budget, IQ2_XS decoded code at
82.3 tok/s and cut off 10 of the 72 answers. The budget removed every cut-off and reduced the average
AIME time from 121 s to 39 s without losing a point. IQ3_XXS scored 2 more MMLU-Pro items, which is
within noise for 42 single samples. It decoded 5-15% slower on most prompts and left only about 11 GB
of RAM free. Strata writes Korean faster; ninfer reads long prompts 2-3x faster and serves four
requests at once, at equal quality.

### With image input enabled (current serving setup)

Both servers now run with image input on (`--vision` for ninfer, Strata's GPU image encoder), which
costs ninfer KV room and Strata some expert-cache VRAM. Same prompts and thinking budget as above.

| Test | ninfer, 27B NVFP4 | Strata IQ2_XS |
|---|---:|---:|
| Greedy code, 1500 tokens | 95.0 tok/s | 80.8 tok/s |
| Korean explanation | 64.6 tok/s | 87.2 tok/s |
| Sampled essay (2 runs) | 78.7 tok/s | 85.2 tok/s |
| First token, 15,840-token prompt | 2.68 s | 8.06 s |
| First token, 31,325-token prompt | 5.73 s | 12.47 s |
| Four parallel 800-token essays, total | 264 tok/s | one request at a time |
| GPU memory per card | 14.8 GB | 15.5 GB |
| System RAM | 24 GB | about 40 GB |
| KV / context | fp8, 114,176 tokens, 98,304 per request | int8, 32,768 resident + RAM streaming, 65,536 |

The quality rows above were measured before image input was enabled. Without the thinking budget,
ninfer cut off 16 of the 72 answers and Strata 10. ninfer's 264 tok/s is with four identical essay
prompts; four different prompts give about 190-200 tok/s.

Which one fits: ninfer for long documents (2-3x faster first token), several requests at once, or a
machine with less RAM. Strata/Swift 1.5 for one Korean conversation at a time (about 35% faster Korean
output) on a machine with 64 GB of RAM or more. The two cannot run side by side on these cards; a small
router can switch engines per request model in about 15-20 s.

### 2026-10-22: Strata v0.2.2, Basalt and ninfer on the same machine

Same machine as above (Ryzen 5 9600X with PBO Auto and Curve Optimizer -40, DDR5 64 GB at
6400 MT/s, 2x RTX 5060 Ti 16 GB on PCIe 5.0 x8/x8 without P2P), now on Ubuntu 26.04.1, NVIDIA
595.91 and CUDA 13.4. Each engine ran alone on both GPUs with one request at a time and temperature 0.
Speed is the second of two runs; each run changed the prompt's first line so no prompt cache was hit.
The 40 verified items are generated arithmetic, date, sorting, counting and remainder questions with
answers computed by the script (exact match on a final `ANSWER:` line), so they compare engines
under one condition and are not a standard benchmark. No run read weights from the SSD.

| | Strata v0.2.2 | [Basalt](https://github.com/jesdga95/basalt) 5790960 | ninfer, this branch 28843fb |
|---|---|---|---|
| Model | Qwen3.8-Flash-Next GSQ-RCO IQ3_S | Qwen3.8-Flash-Next GSQ-RCO IQ3_S | Qwen3.8-27B QUASAR-QAT NVFP4 |
| Context / KV | 262,144 / int8 | 262,144 / int8 | 131,072 / fp8 |
| Prompt, 10,214 tokens | 1,738 tok/s (first token 5.88 s) | 2,326 tok/s (4.39 s) | 6,969 tok/s (1.47 s) |
| Code, 512 tokens | 100.0 tok/s | 116.6 tok/s | 141.5 tok/s |
| Korean, 512 tokens | 74.3 tok/s | 91.7 tok/s | 77.5 tok/s |
| Verified items | 40/40 | 40/40 | 38/40 |
| Tool calls | 5/5 | 5/5 | 5/5 |

ninfer ran the recommended configuration above (`--default-thinking-budget 2048`); Strata used its
installed settings and Basalt its README settings with the same context, KV and image input as Strata,
neither with a thinking budget. ninfer reads prompts 3-4x faster and writes code fastest. It is slower
than Basalt in Korean, and its two misses were the same mistake: rounding a share up instead of down
(126 for 125, 49 for 48). Basalt, a Blackwell-tuned Strata fork, beats Strata on every speed row with
the same model and the same score. One deployment difference: ninfer serves `/health` and the
OpenAI/Anthropic routes but has no load, unload or idle-unload control, while Strata and Basalt can
release the GPUs when idle.

## 2026-10-23 upstream review

Wallawalla47/ninfer-custom was rebased onto a newer upstream after the 2026-10-03 merge, so merging
its `master` again conflicts in 228 files. Its fixes were taken one by one instead (`cherry-pick -x`,
so each commit names its source):

| Commit | Source | What it fixes |
|---|---|---|
| `1513bc5d` | `21ef8232` | NVFP4 A4 epilogue: the scale is rounded before the residual add at every width |
| `b732a674`, `8f614915` | `7a75da46`, `713f8028` | artifact direct reads: aligned staging slots, short `pread`s filled |
| `10b22af7` | `521076f8` | n-gram copy proposals are sized without the output budget |
| `f45ff361` | `3e8f1617` | verification attention partitions KV by the row window, so a round cut by `max_tokens` commits the same KV as a full one (conflict resolved: the RTX 5060 Ti SM-count helper stays; the new test runs without the newer `DeviceExecutionView`) |

Measured with the recommended command on the machine above (DDR5 now at 6400 MT/s), before → after:

| Check | Before (`a226c855`) | After (`f45ff361`) |
|---|---:|---:|
| `ninfer-perplexity --quick`, `--tp 2`, fp8 KV | 4.3865 | 4.3865 |
| 40 generated tasks with computed answers | 38 | 38 |
| Tool calls | 5/5 | 5/5 |
| MTP round time (requests with MTP only) | 26.65 ms | 26.64 ms |
| 10K-token prompt | 6,966 tok/s | 6,935 tok/s |

Single-prompt decode tok/s moved between 95 and 152 from run to run of the same binary (the text and
so the accepted drafts change), so the round time is the speed check here.

Reviewed and not taken, with the reason:

- `d5e6c8f1`, `473df201`, `7ac7c9b4` (split-KV merge as a dependent launch, balanced short decode-row
  splits, TMA BF16 wide `linear_add`): applied cleanly, round time unchanged at 26.64 ms.
- `a9937fc7` (16-bit FP8 residual activations for wide shared rows): aimed at FP8 projections, which
  the all-NVFP4 QUASAR artifact barely has, and its own notes measure 5-6 % less n-gram copy throughput.
- `b278ba57`, `813355b3`, `b2e1aaf6` and the prefill attention rework `d85ee622`: built on frontend and
  attention code this branch does not have (1-52 conflicting files each).
- ValerioDolci/ninfer-tp2 (capy-flash) v0.6.0 overlaps the prefill all-reduces in half-chunks for its
  +60 %; this branch already slices each prefill chunk into up to four column slices whose all-reduce
  overlaps the next slice's work. Its 2048-token chunks are slower here: 2048 and 4096 gave 6,416 and
  6,792 tok/s on the 10K prompt against about 6,950 with 8192.

## Limits

- Two ranks only, for the Qwen3.5-family 27B dense package. MoE layers have no tensor-parallel
  split; the DFlash2 drafter and the Vision encoder are replicated rather than split.
- Rejected at `--tp 2`: `--vision-offload on`, `--prefix-cache-file`, `--spec dflash` (DFlash2
  works), DFlash2 without `--lm-head-draft`, and `--prefill-chunk` above 8192.
- The full test suite passes except tests that need the whole model on one GPU (the `*_real` tests).
  TP1 versus TP2 could not be compared end to end on 16 GB cards. The checks used instead were
  bit-identical ranks and a perplexity match against bf16 KV.
- Numbers come from a single machine and will vary with motherboard, memory and driver.

## License and attribution

This fork stays under the upstream [Apache License 2.0](../LICENSE). Files changed relative to
[Neroued/ninfer](https://github.com/Neroued/ninfer) are listed by `git diff master..tp2-5060ti`.
The TP2 communication approach follows ideas from lynx-gt/ninfer-tp2-5060ti and was reimplemented
here without copying its files.

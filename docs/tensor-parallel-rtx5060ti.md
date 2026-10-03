# Two-GPU tensor parallelism on RTX 5060 Ti

This fork's `tp2-5060ti` branch adds `--tp 2` to NInfer. It splits the Qwen3.8-27B NVFP4 package
across two 16 GB GeForce RTX 5060 Ti cards, so the model fits and runs on cards that cannot share
memory peer-to-peer.

The branch was written by **Claude Opus 5.5 (medium reasoning effort)** in Claude Code. A human
operator set the goals, approved each production change and ran the server. Every commit carries a
`Co-Authored-By: Claude Opus 5.5` trailer.

> 한국어 요약 (2026-10-03 갱신): RTX 5060 Ti 16GB 두 장(P2P 없음)에서 Qwen3.8-27B를 텐서 병렬로
> 서빙하는 브랜치입니다. 모든 코드는 Claude Code의 Claude Opus 5.5가 작성했습니다. 현재 권장 구성은
> QUASAR-QAT 전체 NVFP4 아티팩트 + MTP(드래프트 4) + n-gram 복사 드래프트 + hybrid prefix 캐시입니다. 공식
> 아티팩트 구성 대비 코드 생성 94.9 → 117 tok/s, 한국어 67.6 → 73~85, 1.6만/3.1만 토큰 첫 응답 2.70/5.71 →
> 2.37/5.19초, 짧은 요청 첫 응답 0.10 → 0.06~0.08초, 파일을 거의 그대로 다시 쓰는 편집 128 → 396 tok/s,
> 여러 대화를 번갈아 이어갈 때 첫 응답 0.125 → 0.041초, KV 용량 114K → 223K 토큰입니다. 품질은
> GSM8K 200문항(193~196 대 195)과 MMLU-Pro 210문항(175 대 172)에서 같은 수준입니다. 아래
> "2026-10-03 update"에 측정과 근거가 있고, 그 아래 절들은 공식 아티팩트 시절의 기록입니다.

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

Quality of the recommended configuration: GSM8K 200 (zero-shot, greedy) 193-196 against 195 for the
previous production; MMLU-Pro 210 (15 per category, thinking budget 2048) 175 against 172 for the
official artifact on the same build. Not adopted, with measurements: `nvfp4` KV, earlier A4 routes,
ValerioDolci's pipelined mailbox kernel and wide mailbox slots (our slot is sized for the widest
exchange, so no staged copies remain), ValerioDolci's four-rows-per-warp [5120,8704] GEMV for
T=3..5 (on the RTX 5060 Ti 65.5 → 67.6 µs at T=3/4 and 67.6 → 71.7 µs at T=5, slower than the
sliced-K route kept here), shorter TP timeouts, and a second Wallawalla47 merge (39 commits; equal
speed in an interleaved A/B, kept on branch `tp2-5060ti-walla2`). Benchmarks: `bench/gsm8k_eval.py`, `bench/multiturn.py`, `bench/copy_edit.py`
in the operator's workspace; the commits list their own measurements.

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

Recommended serving configuration (what the measurements below use unless stated otherwise):

```bash
ninfer-serve Qwen3.8-27B-nvfp4-NInfer/qwen3_8_27b_nvfp4.ninfer --host 0.0.0.0 --port 8080 \
  --tp 2 --devices 0,1 \
  --kv-dtype fp8 --max-context 131072 --kv-capacity auto \
  --max-concurrency 4 --device-state-slots 2 --host-kv-mib 16384 --host-state-slots 16 \
  --default-thinking-budget 2048 \
  --prefill-chunk 8192 --spec mtp --draft-tokens 3 --lm-head-draft
```

| Option | Why |
|---|---|
| `--kv-dtype fp8` | Same perplexity as bf16 KV (table above), fastest long-context prefill. |
| `--max-context 131072` | The auto-sized KV pool holds 140,608 tokens, so one request may use up to 131,072 of them. Raising the ceiling from 65,536 did not shrink the pool. |
| `--kv-capacity auto` | Uses the VRAM left after weights and runtime: 140,608 KV tokens instead of 65,536, so more conversations and shared prompts stay cached on the GPU. Leaves 1 GiB headroom (14.8 GB used per card). |
| `--max-concurrency 4` | The 27B weights already sit entirely on the GPUs, so spare memory cannot speed up a single request; it buys parallel requests instead (table below). |
| `--host-kv-mib 16384 --host-state-slots 16` | Pinned host cache for conversations that leave the GPU, doubled from the 8 GiB / 8-slot default. The 64 GB host keeps 34 GB available with the server running. |
| `--default-thinking-budget 2048` | Caps thinking so the answer fits the request's `max_tokens`. At the cap the engine closes thinking and the model answers (table below). Per-request overrides use Anthropic `thinking.budget_tokens`. |
| `--prefill-chunk 8192` | Faster first token on long prompts than 4096 (31,306 tokens 6.2 → 6.0 s, 59,417 tokens 14.0 → 13.7 s, measured before the last attention change); short prompts are unaffected. |
| `--draft-tokens 3 --lm-head-draft` | Best measured MTP setting for single requests; 2 and 4 drafts were slower on average, and turning off the reduced draft head was about 10% slower. With four parallel requests 2, 3 and 4 drafts all reached about 200 tok/s. |

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
Vision. With 131,072 the minimum runtime reservation no longer fits.

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
   output projections are row-split and summed by a residual all-reduce. Norms, embeddings and the
   MTP input projection are replicated.
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
| Host conversation cache | 8 GiB, 8 slots | 16 GiB, 16 slots | 34 GB host memory still free |
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

## Limits

- Two ranks only, for the Qwen3.5-family 27B dense package. MoE layers and DFlash drafts have no
  tensor-parallel split; the Vision encoder is replicated rather than split.
- The full test suite passes except tests that need the whole model on one GPU (the `*_real` tests).
  TP1 versus TP2 could not be compared end to end on 16 GB cards. The checks used instead were
  bit-identical ranks and a perplexity match against bf16 KV.
- Numbers come from a single machine and will vary with motherboard, memory and driver.

## License and attribution

This fork stays under the upstream [Apache License 2.0](../LICENSE). Files changed relative to
[Neroued/ninfer](https://github.com/Neroued/ninfer) are listed by `git diff master..tp2-5060ti`.
The TP2 communication approach follows ideas from lynx-gt/ninfer-tp2-5060ti and was reimplemented
here without copying its files.

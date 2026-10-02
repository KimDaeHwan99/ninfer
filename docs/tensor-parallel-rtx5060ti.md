# Two-GPU tensor parallelism on RTX 5060 Ti

This fork's `tp2-5060ti` branch adds `--tp 2` to NInfer. It splits the Qwen3.8-27B NVFP4 package
across two 16 GB GeForce RTX 5060 Ti cards, so the model fits and runs on cards that cannot share
memory peer-to-peer.

The branch was written by **Claude Opus 5.5 (medium reasoning effort)** in Claude Code. A human
operator set the goals, approved each production change and ran the server. Every commit carries a
`Co-Authored-By: Claude Opus 5.5` trailer.

> 한국어 요약: RTX 5060 Ti 16GB 두 장(P2P 없음)에서 Qwen3.8-27B NVFP4를 텐서 병렬로 서빙하는
> 브랜치입니다. 모든 코드는 Claude Code의 Claude Opus 5.5(Medium)가 작성했습니다. 기준
> 구현(lynx-gt/ninfer-tp2-5060ti) 대비 생성 속도는 약 15% 빠르고, 1.6만 토큰 프롬프트의 첫 응답
> 시간은 5.2초에서 2.7초로 줄었습니다. 권장 설정(동시 처리 4, KV 자동, 생각 상한 2048)에서는 동시
> 요청 4개의 전체 처리량이 264 tok/s(1개일 때의 3.4배)이고, 토큰 한도 4096에서 답이 끊기지 않습니다.

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
  --kv-dtype fp8 --max-context 65536 --kv-capacity auto \
  --max-concurrency 4 --device-state-slots 2 --default-thinking-budget 2048 \
  --prefill-chunk 8192 --spec mtp --draft-tokens 3 --lm-head-draft
```

| Option | Why |
|---|---|
| `--kv-dtype fp8` | Same perplexity as bf16 KV (table above), fastest long-context prefill. |
| `--kv-capacity auto` | Uses the VRAM left after weights and runtime: 140,608 KV tokens instead of 65,536, so more conversations and shared prompts stay cached on the GPU. Leaves 1 GiB headroom (14.8 GB used per card). |
| `--max-concurrency 4` | The 27B weights already sit entirely on the GPUs, so spare memory cannot speed up a single request; it buys parallel requests instead (table below). |
| `--default-thinking-budget 2048` | Caps thinking so the answer fits the request's `max_tokens`. At the cap the engine closes thinking and the model answers (table below). Per-request overrides use Anthropic `thinking.budget_tokens`. |
| `--prefill-chunk 8192` | Faster first token on long prompts than 4096 (31,306 tokens 6.2 → 6.0 s, 59,417 tokens 14.0 → 13.7 s, measured before the last attention change); short prompts are unaffected. |
| `--draft-tokens 3 --lm-head-draft` | Best measured MTP setting for both greedy and sampled chat; 2 and 4 drafts were slower on average, and turning off the reduced draft head was about 10% slower. |

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

## Findings that may help other consumer-GPU setups

- With both GPUs copying at once, each gets about 15.5 GB/s per direction. One GPU alone gets 28.7.
  Host memory bandwidth is the limit, whichever engine (SM loads or copy engines) moves the data.
- Measured `mma.sync` peaks on RTX 5060 Ti: FP16→FP32 54, FP16→FP16 109, E4M3→FP32 109,
  E4M3→FP16 218 TFLOPS.
- Decode reaches about 95% of attainable memory bandwidth. CPU performance governor, disabling PCIe
  ASPM and a 198 W power limit made no measurable difference.

## Comparison with Strata and Swift 1.5 Qwen3.8-Flash-Next

Same machine and prompts. [Strata](https://github.com/Niko1221/Strata) v0.1.32 serves
[Swift 1.5 Qwen3.8-Flash-Next GSQ-RCO IQ2_XS](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
(125B MoE, 6B active, about 2.4 bits per weight). It splits layers across both GPUs, keeps experts in
RAM, uses int8 KV and MTP. It fills both GPUs (15.5 GB) and about 40 GB of RAM, so the two servers
cannot run side by side. ninfer ran with concurrency 1 and no thinking budget for this comparison.

| Test | ninfer, Qwen3.8-27B NVFP4 | Strata, Swift 1.5 IQ2_XS |
|---|---:|---:|
| Greedy code, 1500 tokens | 95.2 tok/s | 82.3 tok/s |
| Sampled essay | 79.0 tok/s | 83.0 tok/s |
| First token, 15,840-token prompt | 2.65 s | 7.64 s |
| First token, 31,325-token prompt | 5.71 s | 12.08 s |
| AIME 2025 #6-15 (14K tokens) | 3/10 | 3/10 |
| MMLU-Pro 42 (4K tokens) | 30/42 | 33/42 |
| HumanEval 20 (4K tokens) | 18/20 | 19/20 |

Swift 1.5 answered with about a third fewer tokens on coding. Strata's 4-item lead on 72 single
samples is within noise; with the thinking budget above, ninfer scored 34/42 and 20/20 on the same
MMLU-Pro and HumanEval items. ninfer reads long prompts 2-3x faster.

## Limits

- Two ranks only, for the Qwen3.5-family 27B dense package. MoE layers have no tensor-parallel split.
- The full test suite passes except tests that need the whole model on one GPU (the `*_real` tests).
  TP1 versus TP2 could not be compared end to end on 16 GB cards. The checks used instead were
  bit-identical ranks and a perplexity match against bf16 KV.
- Numbers come from a single machine and will vary with motherboard, memory and driver.

## License and attribution

This fork stays under the upstream [Apache License 2.0](../LICENSE). Files changed relative to
[Neroued/ninfer](https://github.com/Neroued/ninfer) are listed by `git diff master..tp2-5060ti`.
The TP2 communication approach follows ideas from lynx-gt/ninfer-tp2-5060ti and was reimplemented
here without copying its files.

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
> 시간은 5.2초에서 2.7초로 줄었습니다.

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
instead of one RTX 5090. Build as usual, then convert an official v2 download if needed:

```bash
python3 tools/upgrade_ninfer_v2_to_v3.py qwen3_8_27b_nvfp4.ninfer qwen3_8_27b_nvfp4.v3.ninfer
```

Serve with the configuration used for the results above:

```bash
ninfer-serve qwen3_8_27b_nvfp4.v3.ninfer --host 0.0.0.0 --port 8080 \
  --tp 2 --devices 0,1 \
  --kv-dtype fp8 --max-context 65536 --kv-capacity 65536 --max-concurrency 1 \
  --prefill-chunk 8192 --spec mtp --draft-tokens 3 --lm-head-draft
```

`ninfer` (CLI) and `ninfer-perplexity` accept the same `--tp 2 --devices 0,1`.

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

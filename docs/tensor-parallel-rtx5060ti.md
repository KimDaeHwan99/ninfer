# Two-GPU tensor parallelism on RTX 5060 Ti

This fork's `tp2-5060ti` branch adds `--tp 2` to NInfer. It splits the Qwen3.8-27B NVFP4 package
across two 16 GB GeForce RTX 5060 Ti cards, so the model fits and runs on cards that cannot share
memory peer-to-peer.

The branch was written by **Claude Opus 5.5 (medium reasoning effort)** in Claude Code. A human
operator set the goals, approved each production change and ran the server. Every commit carries a
`Co-Authored-By: Claude Opus 5.5` trailer.

> 한국어 요약: RTX 5060 Ti 16GB 두 장(P2P 없음)에서 Qwen3.8-27B를 텐서 병렬로 서빙하는 브랜치입니다.
> 모든 코드는 Claude Code의 Claude Opus 5.5가 작성했습니다. 권장 구성은 QUASAR-QAT 전체 NVFP4 모델,
> MTP 4토큰 추측, n-gram 복사 추측, hybrid prefix 캐시, fp8 KV, 이미지 입력, VRAM 여유분 512 MiB입니다.
> 2026-10-23 측정값은 다음과 같습니다.
> - 10K 토큰 입력 처리: 7,007 tok/s (첫 토큰 1.46초)
> - 생성: 코드 128~144 tok/s, 한국어 80~84 tok/s
> - 동시 4요청 합계: 337 tok/s
> - KV 용량: 315,008 토큰
> - 정답: GSM8K 97/100, MMLU-Pro 121/140, HumanEval 78/82, KMMLU 76/100
>
> 같은 PC에서 Basalt와 비교한 결과(속도, 지능, 전력, 발열, 메모리)는
> [Basalt vs Ninfer](basalt-vs-ninfer-2x5060ti.md)에 있습니다.

## Recommended configuration

```bash
hf download Feyd89/Qwen3.8-27B-QUASAR-QAT-nvfp4-NInfer qwen3_8_27b_quasar_nvfp4.ninfer --local-dir models
./build/apps/ninfer-serve models/qwen3_8_27b_quasar_nvfp4.ninfer --host 0.0.0.0 --port 8080 \
  --tp 2 --devices 0,1 --kv-dtype fp8 --max-context 131072 --kv-capacity auto \
  --max-concurrency 4 --host-cache-mib 16384 --default-thinking-budget 2048 \
  --prefill-chunk 8192 --spec mtp --draft-tokens 4 --lm-head-draft \
  --ngram-draft-tokens 15 --ngram-min-match 12 --vision --vram-headroom-mib 512
```

| Option | Why |
|---|---|
| QUASAR-QAT artifact | All-NVFP4 weights (quantization-aware training) take 8.85 GiB per card instead of the official artifact's 10.8 GiB, which leaves more room for KV and is faster: code 94.9 → 106 tok/s when it was introduced. |
| `--kv-dtype fp8` | Perplexity equal to bf16 KV (`--quick` corpus with the official artifact: bf16 4.3127, fp8 4.3123, int8 4.3140, nvfp4 4.3188) and the fastest long-prompt prefill. |
| `--max-context 131072 --kv-capacity auto` | The pool is sized from the VRAM left after weights and runtime (315,008 tokens), so one request may use 131,072 and several conversations stay cached. |
| `--max-concurrency 4` | The weights already sit entirely on the GPUs, so spare capacity goes to parallel requests: 337 tok/s aggregate over four requests in the 2026-10-23 evaluation run. |
| `--host-cache-mib 16384` | Pinned host cache for conversations that leave the GPU (8 GiB per rank). |
| `--default-thinking-budget 2048` | Caps thinking so the answer fits `max_tokens`; see [Thinking budget](#thinking-budget). Per-request overrides use Anthropic `thinking.budget_tokens`. |
| `--prefill-chunk 8192` | The widest chunk `--tp 2` accepts. 16384 cost 20 % of the KV pool and deadlocked the two ranks' all-reduce on the first prompt longer than 8192 tokens. |
| `--spec mtp --draft-tokens 4 --lm-head-draft` | Four MTP drafts with the reduced draft head: greedy code 102-106 → 117-120 tok/s over three drafts. Five were no faster on code and slower on Korean and copy edits. |
| `--ngram-draft-tokens 15 --ngram-min-match 12` | Copy drafting for text that repeats the prompt (rewriting a file: 128 → 396 tok/s) at no cost to ordinary generation. |
| `--vision` | Image input. Both GPUs hold a full copy of the encoder (+0.3 GiB each). |
| `--vram-headroom-mib 512` | Measured on this machine; see [Tuning](#tuning---vram-headroom-mib). |

Agent clients that send bursts of requests may also want `--pending-timeout-ms 600000`. With the
default of 30 s, a request that has not started within 30 s of arriving is answered with HTTP 503
`request_queue_timeout` (HTTP 504 `timeout_error` on the Anthropic Messages endpoint).

`ninfer` (CLI) and `ninfer-perplexity` accept the same `--tp 2 --devices 0,1`.

## Measured performance

Measured on 2026-10-23 on the machine below with the recommended command (plus
`--pending-timeout-ms 600000`), build `ffb5ee97`. The full report, with charts, Basalt on the same
machine, power, temperatures and memory: [Basalt vs Ninfer](basalt-vs-ninfer-2x5060ti.md).

| Test | Result |
|---|---:|
| 10,214-token prompt | 7,007 tok/s, first token 1.46 s |
| Code, 512 tokens (three runs) | 128.5 / 143.6 / 140.5 tok/s |
| Korean, 512 tokens (three runs) | 83.6 / 82.6 / 80.2 tok/s |
| GSM8K (100) | 97 |
| MMLU-Pro (10 per category, 140) | 121 |
| HumanEval (82, executed) | 78 |
| KMMLU (100) | 76 |
| 40 generated tasks with computed answers | 38 |
| Tool calls | 5/5 |
| 422 eval items at four concurrent requests | 19.7 min, 337 tok/s aggregate, 220 W average for both GPUs |
| KV pool | 315,008 tokens |

Decode tok/s depends on how many drafts the text accepts, so it moves between runs of the same
binary. The MTP round time (26.6 ms for a four-draft verify) is the steadier hardware figure.

Machine: Ryzen 5 9600X (PBO Enabled, Curve Optimizer −40), 64 GB DDR5 at 6400 MT/s CL30,
two PNY RTX 5060 Ti 16 GB at the default 180 W on PCIe 5.0 x8/x8 without P2P (core +150/+175 MHz,
memory +750 MHz per card), Ubuntu 26.04.1, kernel 7.0, NVIDIA 595.91.07, CUDA 13.4.

## What the branch changes

Each effect was measured on the machine above when the change went in.

1. **TP2 runtime.** Each GPU runs a full program over its weight shard (SPMD). Attention q/gate/k/v,
   GDN q/k/v/z/a/b, MLP gate/up and both output heads are column-split. The attention, GDN and MLP
   output projections are row-split and summed by a residual all-reduce. Norms and the MTP input
   projection are replicated in VRAM; the token embedding is replicated in pinned system RAM
   (see [below](#token-embedding-in-system-ram)).
2. **Collectives over host memory.** Without P2P, every byte crosses host DRAM. Small decode-size
   all-reduces use 8-byte lines that carry 4 payload bytes plus a sequence tag, so one PCIe round trip
   delivers both. They are CUDA Graph capturable (3.7 µs for one 5120-wide column).
3. **Prefill overlap.** Large all-reduces run on copy-engine side streams in up to four column
   slices. Each slice overlaps the next slice's FFN and the next layer's column-wise prologue. This
   cut stream idle time from 17 % to 7.5 % of a prefill.
4. **Kernel routes for shard shapes.** NVFP4 TMA routes and a fused SwiGLU for the
   17408×5120 / 5120×8704 MLP shards (gate/up 2× faster at prefill width). A 24-head GDN control
   MMA (5.6× faster). FP8 split-K waves sized for 36 SMs (4-11 % faster at 1024-2048 tokens).
5. **Device-aware attention.** Causal KV split budgets read the current device instead of assuming
   170 SMs. FP8 prefill attention accumulates PV per key tile in FP16, because consumer Blackwell runs
   FP32-accumulating MMAs at half rate: 16K-key attention 7.01 → 5.22 ms.
6. **NVFP4 rank shards** of the attention/GDN input and [5120,3072] output projections, so
   all-NVFP4 artifacts such as QUASAR-QAT run at `--tp 2`. Shards are tested against the FP64 oracle
   (attention) and bit for bit against the parent (GDN).
7. **Hybrid prefix cache at `--tp 2`.** Both ranks publish Host transfers up to the frontier complete
   on both (deterministic poll), so their prefix indexes stay identical. Three conversations
   alternating over 12K-token documents: resumed-turn first token 0.107 → 0.041 s.
8. **n-gram copy drafting at `--tp 2` with two-width MTP graph families.** Copy rounds verify 16 wide,
   every other round 4 wide on the same frame, so copy drafting no longer slows ordinary generation
   (it cost 10-13 % before the two-width families).
9. **Admission search budget** back to upstream's: the 250 ms search spent 40-128 ms per short
   request once long conversations were cached and found nothing. Short-request first token
   0.19-0.25 → 0.06 s.
10. **Token embedding in pinned system RAM** at `--tp 2` (below).
11. **Link buffers inside the KV plan.** The link's device buffers (a 20 MiB receive scratch and two
    80 MiB partial buffers per card) are created before `--kv-capacity auto` measures free memory,
    so they no longer come out of the headroom, and a small `--vram-headroom-mib` cannot fail startup.
12. **`ninfer-serve` exits with status 2 after an Engine-wide failure**, so a restart policy reloads a
    dead engine.

The branch also carries [Wallawalla47/ninfer-custom](https://github.com/Wallawalla47/ninfer-custom)
(hybrid prefix cache, n-gram copy drafting, programmatic-dependent decode launches, tool-call and
serving fixes, CUDA Graph allowance from measurement) and five later fixes from it (see
[Upstream fixes](#upstream-fixes)). Ideas and one technique (compiling the parent kernels a second
time for a shard) come from [ValerioDolci/ninfer-tp2](https://github.com/ValerioDolci/ninfer-tp2).

## Token embedding in system RAM

`--tp 2` keeps the token embedding table in pinned system RAM by default (`--embedding-offload
auto`, which is `on` at `--tp 2` and `off` on one GPU). The QUASAR artifact stores the table as FP8
rows: 248,320 × 5,120 bytes plus one BF16 scale per row, 1.185 GiB per card. The embedding gather
reads only the rows of the current tokens through the pinned block's mapped address, so the table
never crosses PCIe as a whole.

| Item | Effect |
|---|---|
| KV pool | about +70K tokens from the freed 1.185 GiB per card (estimate, at about 17.5 KB of FP8 KV per token per rank) |
| Decode | each gather reads 5 KiB per token over PCIe, a few microseconds per round of about 27 ms (estimate, under 0.1 %) |
| Prefill | an 8,192-token chunk gathers 42 MB twice with MTP, a few milliseconds per chunk (estimate, well under 1 % of the first token) |
| System RAM | +1.185 GiB pinned per rank, 2.37 GiB in total |
| Output | unchanged: the same kernel reads the same bytes |

Speed with offload on versus off was not A/B tested; every measurement in this guide uses offload.
`--embedding-offload off` restores the VRAM table. `--embedding-offload on` also works on one GPU,
but not together with `--vision-offload on`, whose eviction ladder borrows the device-resident
table. An artifact whose output head shares the embedding's stored object keeps the table in VRAM
for the head. Startup fails with a message if the pinned block is not mapped at the same address on
the device (UVA), which the tensor-parallel link already requires.

## Tuning `--vram-headroom-mib`

`--kv-capacity auto` leaves 1 GiB per card unused by default. How much of it the server really
needs depends on the build and the workload: CUDA loads kernel modules on first use, and Vision,
long prompts and wide n-gram rounds touch more of them. Every 100 MiB moved from the headroom into
the pool is about 6K KV tokens (17.5 KB per token per rank). Measure on the serving machine:

1. Start the server with the recommended command, replacing `--vram-headroom-mib 512` with
   `--vram-headroom-mib 1024` for the default-headroom baseline.
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

Measured on the machine above (the recommended command, build `a9a28d97`):

| `--vram-headroom-mib` | KV pool | Free after start | Minimum free under the tool's load |
|---|---:|---:|---:|
| 1024 (default) | 284,416 tokens | 1,007 MiB | 1,005 MiB |
| 275 (the tool's value) | 329,152 tokens | 259 MiB | 257 MiB |
| 275 with `CUDA_MODULE_LOADING=EAGER` | 318,272 tokens | 275 MiB | - |
| **512 (chosen)** | **315,008 tokens** | 497 MiB | - |

Serving barely allocates after startup here (free memory moved by 2 MiB under the full load), so the
tool recommends 275. Eager module loading takes 10,880 tokens of KV (about 182 MiB), an estimate of
the memory that lazily loaded kernels could still take. Adding that allowance to 275 MiB and
rounding up to 512 MiB keeps the 256 MiB safety margin, for +30,592 KV tokens (+10.8 %) over the
default. The minimum free memory under load was not measured at 512 MiB itself.

## Thinking budget

Without a budget, long thinking used the whole `max_tokens` and the answer was cut off. Measured
when the budget was chosen (official v3 artifact, `max_tokens` 4096, four parallel requests,
temperature 0.6, top_p 0.95, top_k 20, one sample per item):

| `--default-thinking-budget` | MMLU-Pro (42) | HumanEval (20, executed) | Cut off |
|---|---:|---:|---:|
| none | 30 | 18 | 9 |
| 3072 | 33 | 19 | 0 |
| **2048** | **34** | **20** | **0** |
| 1024 | 31 | 20 | 0 |

On AIME 2025 problems 6-15 (`max_tokens` 14000), no budget scored 3/10 with 7 answers cut off and
119 s per problem; a 2048 budget scored 4/10 with none cut off and 51 s per problem. These are small
single-sample sets, so differences of one or two items are within sampling noise. In the
2026-10-23 evaluation (422 items, `max_tokens` 6144) no answer was cut off.

## Image input

`--tp 2` accepts `--vision`. The Vision encoder is not split: both GPUs hold a full copy
(+0.3 GiB each) and encode the same media, so each rank's replicated hidden state stays identical
without a cross-GPU transfer. Checked with the bundled example images (scene description, chart text
and shapes, a two-image difference); all answers matched the images. Repeating the same greedy image
request is not byte-identical, because the second run reuses cached media and prefix state.

## Not adopted

Each was measured on this machine:

- **DFlash2 at `--tp 2`** (replicated drafter) works, but with QUASAR the grafted drafter accepts
  fewer drafts (Korean 12 % against 20 % for MTP) and costs 60 % of the KV pool.
- **`nvfp4` KV**: higher perplexity (table above).
- **ValerioDolci's pipelined mailbox kernel and wide mailbox slots**: this branch's slot is sized
  for the widest exchange, so no staged copies remain.
- **ValerioDolci's four-rows-per-warp [5120,8704] GEMV for T=3..5**: 65.5 → 67.6 µs at T=3/4 and
  67.6 → 71.7 µs at T=5, slower than the sliced-K route kept here.
- **capy-flash v0.6.0 half-chunk prefill overlap** (ValerioDolci/ninfer-tp2): this branch already
  overlaps four column slices per chunk, and its 2048/4096-token chunks gave 6,416/6,792 tok/s on
  a 10K prompt against about 6,950 with 8192.
- **A second whole merge of Wallawalla47/ninfer-custom** (39 commits): equal speed in an interleaved
  A/B; its later fixes were cherry-picked instead (below).
- **Shorter TP timeouts** and **earlier A4 routes**: measured and not adopted.
- **Retraining the MTP head** on this model's own outputs: +15 % draft acceptance offline, but
  within ±1 % in the live server.
- **CPU governor, PCIe ASPM and the GPU power limit**: no measurable effect.

## Upstream fixes

Wallawalla47/ninfer-custom was rebased onto a newer upstream after it was merged here, so merging
its `master` again conflicts in 228 files. Its fixes are taken one by one (`cherry-pick -x`, so each
commit names its source). On this machine a service checks the reference forks after boot, builds
and measures new fixes, and adopts one only if speed and quality do not drop. Taken on 2026-10-23:

| Commit | Source | What it fixes |
|---|---|---|
| `1513bc5d` | `21ef8232` | NVFP4 A4 epilogue: the scale is rounded before the residual add at every width |
| `b732a674`, `8f614915` | `7a75da46`, `713f8028` | artifact direct reads: aligned staging slots, short `pread`s filled |
| `10b22af7` | `521076f8` | n-gram copy proposals are sized without the output budget |
| `f45ff361` | `3e8f1617` | verification attention partitions KV by the row window, so a round cut by `max_tokens` commits the same KV as a full one (conflict resolved: the RTX 5060 Ti SM-count helper stays; the new test runs without the newer `DeviceExecutionView`) |

| Check | Before (`a226c855`) | After (`f45ff361`) |
|---|---:|---:|
| `ninfer-perplexity --quick`, `--tp 2`, fp8 KV | 4.3865 | 4.3865 |
| 40 generated tasks with computed answers | 38 | 38 |
| Tool calls | 5/5 | 5/5 |
| MTP round time | 26.65 ms | 26.64 ms |
| 10K-token prompt | 6,966 tok/s | 6,935 tok/s |

Reviewed and not taken:

- `d5e6c8f1`, `473df201`, `7ac7c9b4` (split-KV merge as a dependent launch, balanced short decode-row
  splits, TMA BF16 wide `linear_add`): applied cleanly, round time unchanged at 26.64 ms.
- `a9937fc7` (16-bit FP8 residual activations for wide shared rows): aimed at FP8 projections, which
  the all-NVFP4 QUASAR artifact barely has, and its own notes measure 5-6 % less n-gram copy throughput.
- `b278ba57`, `813355b3`, `b2e1aaf6` and the prefill attention rework `d85ee622`: built on frontend and
  attention code this branch does not have (1-52 conflicting files each).

## Findings that may help other consumer-GPU setups

- With both GPUs copying at once, each gets about 15.5 GB/s per direction. One GPU alone gets 28.7.
  Host memory bandwidth is the limit, whichever engine (SM loads or copy engines) moves the data.
- Measured `mma.sync` peaks on RTX 5060 Ti: FP16→FP32 54, FP16→FP16 109, E4M3→FP32 109,
  E4M3→FP16 218 TFLOPS.
- Decode reaches about 95 % of attainable memory bandwidth.
- The context-cost model has no RTX 5060 Ti preset, so the generic coefficients apply. They
  overprice recompute by about 1.4× and underprice host transfer by about 2.6× for this machine, but
  restoring a cached 4K-token context (~10 ms) still beats recomputing it (~700 ms) by far more than
  that error, so no cache decision changes.

## Limits

- Two ranks only, for the Qwen3.5-family 27B dense package. MoE layers have no tensor-parallel
  split; the DFlash2 drafter and the Vision encoder are replicated rather than split.
- Rejected at `--tp 2`: `--vision-offload on`, `--prefix-cache-file`, `--spec dflash` (DFlash2
  works), DFlash2 without `--lm-head-draft`, and `--prefill-chunk` above 8192.
- No load, unload or idle-unload control: the server holds both GPUs while it runs.
- The full test suite passes except tests that need the whole model on one GPU (the `*_real` tests).
  TP1 versus TP2 could not be compared end to end on 16 GB cards. The checks used instead were
  bit-identical ranks and a perplexity match against bf16 KV.
- Numbers come from a single machine and will vary with motherboard, memory and driver.

## License and attribution

This fork stays under the upstream [Apache License 2.0](../LICENSE). Files changed relative to the
[Neroued/ninfer](https://github.com/Neroued/ninfer) base `d44ab584` are listed by
`git diff --name-only d44ab584..tp2-5060ti`.
The TP2 communication approach follows ideas from lynx-gt/ninfer-tp2-5060ti and was reimplemented
here without copying its files.

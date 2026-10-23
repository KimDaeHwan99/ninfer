# NInfer 두 GPU 포크 (`tp2-5060ti`) 한국어 안내

이 저장소의 `tp2-5060ti` 브랜치는 [NInfer](https://github.com/Neroued/ninfer)를 **RTX 5060 Ti 16GB 두 장**에서
돌릴 수 있게 고친 포크입니다. 원본 NInfer는 RTX 5090 한 장(32GB)용이라, 27B 모델이 16GB 카드 한 장에는
들어가지 않습니다. 이 포크는 모델을 두 GPU에 나눠 싣는 텐서 병렬(`--tp 2`)을 더해서, 두 카드 사이에
P2P가 안 되는 일반 소비자용 PC에서도 Qwen3.8-27B를 빠르게 돌립니다.

포크의 코드는 Claude Code의 Claude Opus 5.5가 작성했고, 사람 운영자가 목표를 정하고 변경을 승인했습니다.

- 영어 상세 문서: [Two-GPU tensor parallelism on RTX 5060 Ti](docs/tensor-parallel-rtx5060ti.md)
- 같은 PC에서 Basalt와 비교한 벤치마크: [Basalt vs Ninfer](docs/basalt-vs-ninfer-2x5060ti.md)

## 이런 분께 맞습니다

- **RTX 50 시리즈(Blackwell, `sm_120a`) 카드 두 장**으로 로컬 LLM을 돌리려는 분. 실제로 시험한 카드는 RTX 5060 Ti 16GB 두 장입니다.
- 긴 문서나 코드를 넣고 빨리 답을 받아야 하는 분. 10K 토큰 입력을 1.5초 안에 처리합니다.
- 여러 요청을 동시에 처리해야 하는 분. 에이전트, 여러 사용자, 일괄 평가 등이 해당합니다.
- 시스템 RAM이 많지 않은 분. 모델이 17.4GB라 VRAM에 다 들어가고, 시스템 RAM은 약 23GB만 씁니다(대화 캐시 16GB 포함).

## 측정 결과 (2026-10-23)

| 항목 | 결과 |
|---|---:|
| 10K 토큰 입력 처리 | 7,007 tok/s (첫 토큰 1.46초) |
| 코드 생성 | 128~144 tok/s |
| 한국어 생성 | 80~84 tok/s |
| 동시 4요청 합계 | 337 tok/s |
| 대화 기억 용량 (KV) | 315,008 토큰 |
| GSM8K / MMLU-Pro / HumanEval / KMMLU | 97/100 · 121/140 · 78/82 · 76/100 |

측정에 쓴 PC는 Ryzen 5 9600X, DDR5 64GB(6400 MT/s), RTX 5060 Ti 16GB × 2(PCIe 5.0 x8/x8, P2P 없음),
Ubuntu 26.04.1, NVIDIA 드라이버 595.91, CUDA 13.4입니다. 생성 속도는 추측 디코딩이 맞는 비율에 따라
매번 조금씩 달라집니다.

## 준비물

- 64비트 Linux (Ubuntu 24.04와 26.04.1에서 확인. Windows의 WSL2도 가능)
- `sm_120a` GPU 두 장과 NVIDIA 드라이버 580 이상
- CUDA 13 툴킷 (13.4 권장), CMake 3.28 이상, C++20 컴파일러, Ninja, `pkg-config`
- FFmpeg와 curl 개발 패키지
- 저장 공간: 모델 파일 17.4GB, 빌드 폴더 몇 GB

## 빌드

```bash
sudo apt-get install -y build-essential cmake ninja-build pkg-config \
  libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libcurl4-openssl-dev
git clone --branch tp2-5060ti https://github.com/KimDaeHwan99/ninfer.git
cd ninfer
cmake --preset release
cmake --build build -j
```

빌드가 끝나면 서버 프로그램은 `build/apps/ninfer-serve`에 있습니다.

`src/core/nvtx.h`에서 빌드가 멈추면 CUDA NVTX 헤더가 없는 것입니다. 툴킷과 같은 버전의 NVTX 패키지를 설치하세요(CUDA 13.4라면 `cuda-nvtx-13-4`).

## 모델 받기

권장 모델은 양자화 인식 학습(QAT)을 거친 Qwen3.8-27B QUASAR NVFP4입니다(17.4GB).

```bash
hf download Feyd89/Qwen3.8-27B-QUASAR-QAT-nvfp4-NInfer qwen3_8_27b_quasar_nvfp4.ninfer --local-dir models
```

## 실행 (권장 설정)

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_quasar_nvfp4.ninfer --host 0.0.0.0 --port 8080 \
  --tp 2 --devices 0,1 --kv-dtype fp8 --max-context 131072 --kv-capacity auto \
  --max-concurrency 4 --host-cache-mib 16384 --default-thinking-budget 2048 \
  --prefill-chunk 8192 --spec mtp --draft-tokens 4 --lm-head-draft \
  --ngram-draft-tokens 15 --ngram-min-match 12 --vision --vram-headroom-mib 512
```

같은 PC 안에서만 쓸 거라면 `--host 0.0.0.0` 대신 `--host 127.0.0.1`을 쓰세요. 다른 컴퓨터에서 접속하게
열어 둘 때는 `--api-key`로 키를 거는 것이 안전합니다.

| 옵션 | 뜻 |
|---|---|
| `--tp 2 --devices 0,1` | 0번과 1번 GPU에 모델을 나눠 싣습니다 |
| `--kv-dtype fp8` | 대화 기억(KV)을 fp8로 저장합니다. bf16과 정확도가 같고 메모리는 절반입니다 |
| `--max-context 131072` | 요청 하나가 쓸 수 있는 최대 길이(약 13만 토큰)입니다 |
| `--kv-capacity auto` | 남는 VRAM을 모두 대화 기억에 씁니다 |
| `--max-concurrency 4` | 요청 4개를 동시에 처리합니다 |
| `--host-cache-mib 16384` | GPU에서 밀려난 대화를 시스템 RAM에 16GB까지 보관합니다. 대화를 다시 이어갈 때 빠릅니다 |
| `--default-thinking-budget 2048` | 생각(추론) 길이를 2,048 토큰으로 제한합니다. 생각이 길어져 답이 잘리는 것을 막습니다 |
| `--prefill-chunk 8192` | 입력을 8,192 토큰씩 나눠 처리합니다. 두 GPU 모드에서 쓸 수 있는 최대값입니다 |
| `--spec mtp --draft-tokens 4 --lm-head-draft` | 다음 4토큰을 미리 추측하고 한 번에 검증해서 생성을 빠르게 합니다 |
| `--ngram-draft-tokens 15 --ngram-min-match 12` | 입력에 있던 글을 그대로 옮겨 쓸 때(파일 수정 등) 한 번에 여러 토큰을 씁니다 |
| `--vision` | 이미지 입력을 켭니다 |
| `--vram-headroom-mib 512` | VRAM 여유분을 512MiB만 남기고 나머지를 대화 기억에 씁니다. 기본값은 1GiB입니다 |

에이전트처럼 요청을 한꺼번에 많이 보내는 프로그램을 붙일 때는 `--pending-timeout-ms 600000`도 더하세요.
기본값(30초) 안에 차례가 오지 않은 요청은 HTTP 503 오류로 거절됩니다.

## 써 보기

서버는 OpenAI 호환 API(Chat Completions, Responses)와 Anthropic Messages API를 제공합니다.
Open WebUI, Continue, Claude Code 같은 프로그램에 `http://<서버 주소>:8080`을 연결하면 됩니다.

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "<모델 이름>", "messages": [{"role": "user", "content": "안녕하세요. 자기소개를 해 주세요."}]}'
```

`<모델 이름>`에는 `/v1/models`가 알려 주는 이름을 넣으세요. 기본값은 불러온 모델의 이름이고,
`--model-id`로 원하는 이름을 붙일 수 있습니다.

## VRAM 여유분 맞추기

`--vram-headroom-mib 512`는 위 PC에서 측정한 값입니다. 다른 PC나 다른 빌드에서는 직접 재 보는 것이 좋습니다.
여유분이 너무 작으면 실행 중 `CUDA_CHECK(...) failed`를 내며 서버가 멈추고, 진행 중이던 요청과 저장된 대화가 사라집니다.

1. `--vram-headroom-mib 1024`로 서버를 띄웁니다.
2. 저장소에서 측정 도구를 돌립니다. 여러 종류의 부하를 걸면서 실제로 남는 VRAM을 잽니다.

   ```bash
   python3 tools/bench/measure_vram_headroom.py --base-url http://127.0.0.1:8080 \
     --gpus 0,1 --headroom-mib 1024 --output profiles/bench/headroom
   ```

3. 도구가 알려 준 값으로 다시 띄우고 한 번 더 재서, 안전 여유(256MiB)가 남는지 확인합니다.

자세한 방법은 영어 문서의 [Tuning 절](docs/tensor-parallel-rtx5060ti.md#tuning---vram-headroom-mib)에 있습니다.

## 자주 겪는 문제

| 증상 | 해결 |
|---|---|
| 빌드가 `src/core/nvtx.h`에서 멈춤 | `cuda-nvtx-13-x` 패키지 설치 (툴킷과 같은 버전) |
| 요청이 HTTP 503 `request_queue_timeout`으로 거절됨 | `--pending-timeout-ms 600000` 추가 |
| 실행 중 `CUDA_CHECK(...) failed` 후 서버 종료 | `--vram-headroom-mib`를 늘리고 위 측정 도구로 다시 확인 |
| `--prefill-chunk 16384`가 거절됨 | 두 GPU 모드에서는 8192가 최대입니다 |
| WSL2에서 모델 로딩이 느림 | `.ninfer` 파일을 `/mnt/c/...`가 아니라 Linux 쪽 폴더로 옮기기 |
| WSL2에 CUDA 13.4 설치 | Linux용 NVIDIA 드라이버는 설치하지 말고, `ubuntu2404` 저장소에서 `cuda-toolkit-13-4`만 설치 |

## 한계

- 두 GPU까지만 지원하고, Qwen3.5 계열 27B Dense 모델용입니다. MoE 모델은 나눠 실을 수 없습니다.
- 서버를 켜 두는 동안 두 GPU를 계속 차지합니다. 쓰지 않을 때 GPU를 자동으로 비우는 기능은 없습니다.
- 위 수치는 PC 한 대의 결과입니다. 메인보드, 메모리, 드라이버에 따라 달라집니다.

## 이 포크가 바꾼 것

- **두 GPU 분할**: 가중치와 계산을 두 GPU에 나눕니다.
- **P2P 없는 통신**: GPU 사이 통신을 시스템 RAM의 고정 메모리를 거쳐 하도록 만들었습니다.
- **입력 처리 중 통신 겹치기**: 계산과 통신을 동시에 진행합니다.
- **카드에 맞춘 커널**: 조각 크기와 36 SM에 맞춘 연산 경로를 넣었습니다.
- **소비자용 Blackwell 맞춤 어텐션**: 해당 카드 구조에 맞게 어텐션을 조정했습니다.
- **추측 디코딩과 대화 캐시를 두 GPU에서 사용**: MTP, n-gram 추측, 대화 캐시가 두 GPU 모드에서도 동작합니다.
- **임베딩 표를 RAM으로**: 임베딩을 시스템 RAM으로 옮겨 VRAM을 대화 기억에 더 씁니다.

기술별 설명과 효과 수치는 [Basalt vs Ninfer](docs/basalt-vs-ninfer-2x5060ti.md#두-엔진과-적용한-커스텀)와
[영어 문서](docs/tensor-parallel-rtx5060ti.md#what-the-branch-changes)에 있습니다.

## 라이선스

원본과 같은 [Apache License 2.0](LICENSE)입니다. 두 GPU 통신 방식은 lynx-gt/ninfer-tp2-5060ti의 아이디어를
참고해 새로 구현했고, Wallawalla47/ninfer-custom과 ValerioDolci/ninfer-tp2의 작업을 함께 담고 있습니다.

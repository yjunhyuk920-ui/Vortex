# 원본 반올림 트리의 부호 궤도: 조건부 native 참조 구성

2026-09-07 Asia/Seoul. Parent: PR #131, `8262178842ec4699a45093f3c29a669cfc158481`.

## 결과와 적용 범위

작은 잔차, 낮은 rank, 작은 입력 변화 없이 원본 계산 subtree와 가중치 부호를 함께 뒤집은 subtree를 공유하는 생성기·직렬화·Python 참조기·최적화 C 실행기를 구성했다. 입력/가중치는 유한 BF16이고 일반 가수·지수·부호 및 subnormal을 포함한다. ABI는 별도 FP32 곱셈, 인접 균형 FP32 합산 트리, 마지막 BF16 RNE로 고정했다. 원본 Hugging Face/CUDA/Tensor Core/FMA의 ABI를 보존했다는 뜻은 아니다.

**THEORY_STATUS=NOT_ESTABLISHED; CORE_ADMISSION=false; HARDWARE_STATUS=NOT_TESTED; FULL_MISSION_O1_O6=OPEN; THREE_QUALIFYING_NEW_PRINCIPLES=false.**

## 실행 규칙과 정확성

각 subtree E를 `(eval_native(E), eval_native(mirror(E)))`로 표현한다. mirror는 모든 가중치의 부호를 반전한 동일한 계산 트리다. 잎은 입력 열과 BF16 magnitude로 intern한다. 자식 tagged refs가 `(l,r)`이면 `p=l&1`로 두 phase를 동시에 정규화한 `(l XOR p,r XOR p)`를 key로 공유한다. 자식 순서·입력 열·괄호를 바꾸지 않는다. 구조 귀납법으로 각 슬롯이 해당 원본 평가와 일치한다. 원본 W 배열은 질의 객체에 없지만 가중치 정보는 코드와 주소에 남으며 전부 계상한다.

단순 부호 반전은 signed zero에서 틀린다. `RN32(1-1)=+0`, mirrored sum도 `+0`이지만 첫 결과를 negate하면 `-0`이다. C 최적화는 finite nonzero 결과에만 sign flip을 적용한다. 합산 결과가 0이면 mirrored children이 둘 다 -0인 경우에만 -0, 나머지는 +0을 저장한다. nonfinite intermediate에서는 mirrored native add를 실제 수행한다. 유한 입력의 잎 곱셈은 부호 대칭을 사용한다. FE_TONEAREST, gradual underflow, nontrapping pure primitives를 전제하고 FP status flags는 ABI 밖이다. NaN 입력/가중치는 거부하며 중간 NaN의 payload는 고정 CPU 연산에만 해당한다.

이는 CSE/구조화 변환의 한 참조 구현이지 새로운 학술 분야나 범용 고속 엔진의 발견이 아니다. 빠른 Walsh 변환 자체는 선행연구다.

## 실제 결과

초기 24개 행렬 중 22개 실행, 일반 BF16 256폭 두 개는 가중치 magnitude 잎 비용으로 생성 전에 제외했다. 초기 Python의 176개 질의/54,272개 출력 FP32·BF16 불일치 0. 독립 정수 IEEE oracle 16,000개 primitive와 72개 작은 전체 출력도 일치했다.

후속 C는 같은 초기 입력에 별도 사전등록한 2048폭 4개 집단을 추가했다. **192개 질의/87,040개 출력 FP32·BF16 불일치 0**. 초기 표본을 독립 표본처럼 다시 더하지 않는다. 이 입력들의 출력은 유한했고, 별도 4개 적대 사례에서 ±0, underflow, overflow/NaN, 실제 rounding cancellation을 검사했다. 200개 작은 C fuzz와 24-step 비-Transformer state/RNG smoke는 단위검사에 포함된다.

| seed7 집단 | 실행 코드/원본 BF16 | 최대 논리적 코드+작업자료 읽기·쓰기/원본 weight payload | FP 산술/원본 |
|---|---:|---:|---:|
| Walsh 1024 | 4.4937% | 11.2371% | 0.5374% |
| Walsh 2048 | 2.4418% | 6.1079% | 0.2930% |
| 열 순열 Walsh 256 | 75.8057% | 189.5630% | 9.3933% |
| 임의 부호 256 | 77.0203% | 193.2037% | 9.5455% |
| 일반 BF16 64, 진단용 | 782.2266% | 1713.7207% | 98.1053% |

Walsh와 row-signed-permuted Walsh는 의도적으로 구조화한 양성 대조군이다. 모두 조밀·full-rank지만 공개 학습 모델은 아니다. 일반 가중치에서 같은 이득은 확보하지 못했다. 읽기 숫자는 주소·입력 검사·scratch·출력을 포함한 논리 계측이지 cacheline/DRAM/GPU 트래픽이나 latency 측정이 아니다. 보수적 word-work envelope `32L+64A+16m+8n`까지 더한 2048 비율은 약 18.85%라 전체 작업 10배 절감을 인증하지 않았다. 이 상한이 높다고 실제 지연의 하한이 증명되는 것도 아니다.

## 비용과 제외 근거

SORB1 코드 바이트는 `32+8(L+A)+4m`. 각 열의 서로 다른 가중치 magnitude 수의 합이 L이다. 잎만으로 코드<=원본의 10%를 만족하려면 `L/(mn)<=0.025`가 필요하다. 이 형식만의 필요조건이며 전역 불가능성 정리가 아니다. 구조화 n-square 계열은 `L=n,A=n log2(n)`이지만 임의 W에서는 `L<=mn,A<=m(n-1)`로 돌아간다.

초기의 `A=n log2(n)/2` 예상식은 틀렸고 테스트 실패를 보존했다. 수정식은 `A=n log2(n)`이다. 후속 최적화/2048 입력은 초기 결과 이후 별도로 사전등록했다.

현재 생성기는 W 전체와 refs를 RAM에 보유한다. 원본·코드 저장, 여러 사전검사/생성 스캔, dict 비용, 파일 로딩 복사, 할당 비용이 남는다. 코드의 `preparation_weight_reads=mn`은 leaf 구성 pass만 가리키며 전체 준비 비용이 아니다. 417,824B라는 2048 core-buffer 합은 Python/allocator/loader 등을 제외한 자료구조 크기다. 8GiB 전체 peak나 405B scale 검증으로 사용하지 않는다. uint32 주소/600,000 node cap을 넘으면 중단하며 fallback 성공으로 세지 않는다.

## 비교한 다른 원리와 미해결 의무

선형 좌표계에 상태를 남기는 방향에서, 실수 전체에 `SiLU(Ax)=B SiLU(x)+Cx`를 요구하는 단순 형식은 invertible A가 signed permutation이어야 한다. Taylor 2차 mixed term이 한 행에 두 비영 계수를 금지하고, 2차·4차 비교가 `b=a²=a⁴`, 따라서 `a=±1`을 준다. B=|A|, C=(A-|A|)/2. 더 일반적인 conjugated gate나 finite-word 방식 전체를 배제하지 않는다. 출력 인덱스 recurrence 방향은 arbitrary W의 저비용 생성기를 구성하지 못해 큰 백엔드를 만들지 않았다.

O1 일반 checkpoint의 작은 표현, O2 전체 prefill/decode/state, O3 실제 원본 kernel/RNG/state, O4 전체 준비/실행 비용, O5 405B/8GiB/4BQ4/TTFT, O6 전체 증명은 OPEN이다. pure kernel의 동일 비트 반환을 전제로 한 state composition은 조건부 보조 정리다. 공개 모델·Transformer KV·GPU·405B·baseline·전체 저장소 테스트·Actions는 실행하지 않았다.

## 재현과 보존

20개 로컬 단위검사 통과. 188개 manifest 지정 결과/실행 파일과 두 manifest를 재생성해 바이트 일치. capsule 복원 후 C 재빌드와 20개 검사도 통과했다.

```bash
python experiments/signed_orbit_20260907/restore.py --out /tmp/vortex-signed-orbit
cd /tmp/vortex-signed-orbit
cc -O2 -std=c11 -fPIC -shared -fno-fast-math -ffp-contract=off -frounding-math src/native.c -lm -o src/native.so
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 python src/run.py --out regenerated/run
OPENBLAS_NUM_THREADS=1 python src/followon.py --old regenerated/run --out regenerated/followon --lib src/native.so
# From the repository, use the original restore.py path:
python /path/to/Vortex/experiments/signed_orbit_20260907/restore.py --verify /tmp/vortex-signed-orbit/regenerated
```

34 text files restore exactly: all Python/C sources, tests, original/follow-on preregistrations, source hashes, full Korean proofs/cost report, raw per-query JSON observations, environment/compiler versions, initial failed and final passed logs, and binary-file SHA256 manifests. Binary weights/inputs/programs are **regenerated and compared with recorded hashes**, not embedded wholesale in these capsules. Full binary evidence also exists in the local research bundle. XZ/base64 here is archival encoding, not an inference compression result. Payload SHA256: `de85f7342d50bde12ee2de39515e8b60f9c0a455eb9e10c8445ad42f43f21257`.

Next research must construct a coefficient-magnitude-agnostic cheap causal information source, not raise DAG caps, replace the checkpoint by Walsh/low-bit weights, or rename CSE as a universal engine. Native orbit rules remain auxiliary. Prior evidence/policies remain unchanged. Remote persistence is independent from scientific success.

Sources: NVIDIA floating-point guide https://docs.nvidia.com/cuda/floating-point/index.html ; Alman arXiv:2211.04643 https://arxiv.org/abs/2211.04643 ; PyTorch SiLU definition https://docs.pytorch.org/docs/stable/generated/torch.nn.SiLU.html . All experimental quantities above are from the archived local records, not those sources.

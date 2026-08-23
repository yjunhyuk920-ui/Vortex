# VORTEX fixed mission and working principles

This is the canonical compact answer to **“우리의 목표와 작업 원칙은?”**. Every session reads it immediately after `AGENTS.md`.

## 고정 목표

\[
\boxed{\text{임의의 공개·미수정 Hugging Face Dense 405B급 모델을}}
\]
\[
\boxed{\text{재학습·미세조정·증류·가중치 수정 없이 실행기만 교체하여}}
\]
\[
\boxed{\text{단일 8 GiB GPU에서 원본 출력·successor-state 계약을 보존하면서}}
\]
\[
\boxed{\text{같은 머신의 native 4B Q4에 가까운 지연으로 실행한다.}}
\]

최종 성공선:

\[
p50 \le 1.2\times T_{\mathrm{native\ 4B\ Q4}},\qquad
p95 \le 1.5\times T_{\mathrm{native\ 4B\ Q4}}.
\]

## 절대 원칙

- 임의의 공개 Dense checkpoint를 대상으로 한다. 특정 prompt·row·layer·checkpoint의 우연을 보편 해법으로 승격하지 않는다.
- retraining, fine-tuning, distillation, LoRA, 의미를 바꾸는 pruning·근사 weight 수정으로 목표를 회피하지 않는다.
- 선언된 ABI/RNG 조건에서 token뿐 아니라 필요한 successor state까지 원본 계약을 보존한다.
- 실패 시 원본 405B 전체 재실행이 필요하면 fallback 비용을 숨기지 않고 전부 계상한다.
- GPU VRAM, CPU RAM, SSD, PCIe, HBM, KV, sidecar, metadata, decompression, packing, verification, repair, synchronization, candidate 수 \(N\), commit token 수 \(A\)를 모두 계상한다.
- 원격 또는 숨은 대형 연산을 사용하고 단일 8 GiB 실행이라고 부르지 않는다.
- `MEASURED`, `DERIVED`, `PROJECTED`, `UNVERIFIED`를 분리하고 측정하지 않은 값은 `NOT TESTED`로 남긴다.
- 결정적으로 탈락한 방법을 threshold, seed, rank, tile, block length만 바꾸어 반복하지 않는다.
- 핵심 연구 회차마다 숨은 전제·계산 순서·정보 흐름·검증 단위 중 하나 이상을 뒤집는 서로 다른 새 원리 3개를 먼저 만든다.
- 그중 원본 dense arithmetic 또는 weight movement를 최소 10× 줄일 수 있는 명시적 경로와 가장 싼 반증 Gate가 있는 후보만 구현한다.
- `REAL_EXECUTOR_ONLY`가 authoritative arm이다. 미래 target token/state, perfect selector, 무료 `N/A`, 무료 transform/workspace/repair/fallback, 미측정 압축률은 promotion 근거가 될 수 없다.
- README는 선택 사항이 아니다. 목표, 작업 방식, 최신 권위 결과, 활성 frontier, 실행법, 저장소 구조 중 하나가 바뀌면 같은 회차에 갱신한다.

## 연구·검증·GitHub 운영 원칙

\[
\boxed{\textbf{연구와 검증은 로컬/샌드박스에서 끝낸다. GitHub는 최종 커밋과 인계에만 사용한다.}}
\]

기본 연구 흐름은 다음으로 고정한다.

```text
LOCAL_RESEARCH
-> LOCAL_VALIDATION_PASS
-> COMMIT_PUSHED
-> REMOTE_COMMIT_VERIFIED
```

### 1. `LOCAL_RESEARCH`

로컬/샌드박스에서 다음을 반복한다.

- 수식·정확 산술·비용식·roofline 계산;
- tensor/circuit/program/combinatorial search;
- prototype과 빠른 수정;
- exhaustive/random/adversarial controls;
- unit/property/regression tests;
- 사용 가능한 공개 checkpoint의 실제 실행;
- raw evidence와 checksum 생성.

### 2. `LOCAL_VALIDATION_PASS`

GitHub로 보내기 전에 로컬에서 해당 회차에 필요한 검증을 완료한다.

- exact token/output/state 계약;
- causal information availability;
- 실제 구현 경로의 arithmetic/traffic/cache/workspace/fallback 비용;
- 8 GiB 및 target-scale resource equation;
- 가장 강한 negative control;
- 결과 재생성 및 checksum;
- 변경된 코드의 focused/full test 중 가능한 범위.

로컬 환경에서 수행하지 못한 항목은 통과시킨 것으로 간주하지 않고 `NOT TESTED`로 기록한다.

### 3. `COMMIT_PUSHED`

로컬 검증이 끝났으면 GitHub에서는 **같은 연구를 다시 실행하지 않는다**. 다음을 한 번에 보존한다.

- source/config/dependency pin;
- raw/processed result;
- logs/checksums;
- scientific decision와 claim boundary;
- 바뀐 canonical ledgers;
- 최신 README.

### 4. `REMOTE_COMMIT_VERIFIED`

원격 branch HEAD와 commit SHA를 다시 읽어 실제로 저장됐는지만 확인한다. 이것으로 일반 연구 회차의 GitHub 단계는 끝난다.

## GitHub Actions 정책

로컬에서 연구와 검증이 완료된 회차에 대해 **GitHub Actions 재실행, clean-room reproduction, hosted checkpoint rerun, 별도 artifact reproduction은 요구하지 않는다.**

기본값:

```text
GITHUB_REEXECUTION_REQUIRED=false
GITHUB_ACTIONS_REQUIRED=false
```

GitHub Actions는 사용자가 명시적으로 요청한 경우에만 예외적으로 사용한다. CI가 이미 자동으로 존재하더라도 그 실행 결과는 로컬에서 검증 완료된 과학적 판정을 다시 승인하는 필수 Gate가 아니다.

즉 다음은 금지한다.

```text
local validation PASS
-> 같은 실험을 GitHub Actions에서 또 실행
-> Actions 결과를 기다려야 연구 완료라고 판정
```

대신:

```text
local validation PASS
-> 결과 포함 commit/push
-> remote SHA read-back
-> done
```

## Core resource objective

최소 accounting frame:

\[
T_{\rm token}\ge\max\left(\frac{S_c}{BA},\;r\frac{N}{A}\frac{2P}{F}\right).
\]

따라서 credible core는 동시에

\[
A\gg1,\qquad N/A\rightarrow1,\qquad r\ll1
\]

을 만족해야 한다. Accepted length만 늘고 \(r=1\)이면 core 성과가 아니다.

## README freshness contract

매 의미 있는 repository 회차에서 `README.md`를 확인한다. mission, governance, 최신 authoritative result, active frontier, quick start, dependency, repository map, advertised capability/test count가 바뀌면 같은 commit에 반영한다.

변경 시:

```text
README_CURRENT=true
README_UPDATED=<sections>
```

변경 불필요 시:

```text
README_CURRENT=true
README_UNCHANGED_REASON=<specific reason>
```

## 상태 용어

```text
LOCAL_RESEARCH
LOCAL_VALIDATION_PASS
LOCAL_VALIDATION_FAILED
COMMIT_PUSHED
REMOTE_COMMIT_VERIFIED
BLOCKED_REMOTE_WRITE
NOT_TESTED
```

`WORKFLOW_RUNNING`, `WORKFLOW_FAILED`는 사용자가 별도 GitHub Actions 실행을 요청한 예외 회차에서만 사용한다.

## Authoritative source order

1. 원격 branch/commit metadata;
2. commit된 raw result/checksum/log;
3. `RESEARCH_STATE.md`, `NEXT_EXPERIMENT.md`, canonical ledgers;
4. 이 문서와 `AGENTS.md`;
5. `README.md`;
6. conversation memory.

Conversation memory는 authoritative research state가 아니다.

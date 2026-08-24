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
- `REAL_EXECUTOR_ONLY`가 authoritative arm이다. 미래 target token/state, perfect selector, 무료 `N/A`, 무료 transform/workspace/repair/fallback, 미측정 압축률은 promotion 근거가 될 수 없다.
- README는 선택 사항이 아니다. 목표, 작업 방식, 최신 권위 결과, 활성 frontier, 실행법, 저장소 구조 중 하나가 바뀌면 같은 회차에 갱신한다.

## 연구 후보 선택 원칙 — 숙제식 실험 금지

새 EXP 번호를 만들거나 코드를 구현하기 전에 반드시 **기술적 사전판단**을 한다. 아이디어를 하나 만들었다는 이유만으로 실험하지 않는다.

핵심 회차마다 서로 다른 원리 3개를 만들되, 각 후보에 대해 결과를 보기 전에 다음을 기록한다.

```text
PRIOR=<HIGH|MEDIUM|LOW>
WHY_IT_MIGHT_WORK=<작동할 구조적 이유>
WHY_IT_MIGHT_FAIL=<가장 강한 반론/반례>
DOMINANT_TERM_CHANGED=<실제 비용식에서 무엇이 사라지거나 크게 줄어드는가>
MAX_IMPACT_IF_TRUE=<성공 시 최대 파괴력>
ARBITRARY_CHECKPOINT_ARGUMENT=<임의 dense checkpoint에서도 버틸 근거>
CHEAPEST_KILL=<가장 싼 결정적 반증>
FALSIFICATION_COST=<LOW|MEDIUM|HIGH>
IMPLEMENTATION_COST=<LOW|MEDIUM|HIGH>
DECISION=<GO|NO_GO|CHEAP_KILL_ONLY>
```

`PRIOR`는 측정된 확률이 아니라, 지금까지의 실패 기록·자원식·정보 흐름·대수 구조·강한 반례를 반영한 기술적 사전확률 판단이다. 결과를 본 뒤 사후적으로 바꾸지 않는다.

후보의 연구 우선순위는 다음 직관으로 정한다.

\[
\boxed{
\text{Research Value}\propto
\frac{P(\text{works})\times\text{impact if true}}
{\text{cost to falsify}}
}
\]

이 식은 순위 휴리스틱이며 가짜 수치확률을 만들라는 뜻이 아니다.

### 구현 전 즉시 `NO_GO` 하는 경우

- 완벽히 성공해도 첫 `10x` core reduction에 못 미친다.
- dominant cost는 그대로인데 보조항만 조금 줄인다.
- 간단한 합법적 arbitrary-dense 반례가 이미 universal claim을 깨뜨린다.
- 미래 target 정보, 무료 계산, 숨은 compute, 미측정 압축률이 필요하다.
- 특정 모델의 low-rank/sparsity/repetition 우연에만 기대고 arbitrary checkpoint 근거가 없다.
- 이미 닫힌 family의 parameter sweep·이름변경·근접 변형이다.
- 가장 싼 자원식만으로 8 GiB 또는 p50/p95 목표를 결정적으로 넘는다.

5줄짜리 식으로 죽는 아이디어에 500줄의 코드를 쓰지 않는다.

### `CHEAP_KILL_ONLY` 문샷 예외

성공 확률이 낮아 보여도 **성공하면 지배항 자체를 없애고**, 반증이 매우 싸다면 최소 Gate만 먼저 실행할 수 있다. 예를 들어 한 weight sweep으로 여러 exact causal state/token을 전진시키거나, 원본 dense sweep을 제거하거나, \(r\)을 order-of-magnitude로 줄이는 원리는 낮은 prior라도 연구 가치가 있다.

반대로 성공 가능성이 높더라도 10~20% 수준의 작은 개선만 예상되면 현재 VORTEX core에서는 우선순위를 낮춘다.

상세 규칙은 [`docs/research/RESEARCH_PRIORITIZATION_CONTRACT.md`](docs/research/RESEARCH_PRIORITIZATION_CONTRACT.md)를 따른다.

## 연구·검증·GitHub 운영 원칙

\[
\boxed{\textbf{연구와 검증은 로컬/샌드박스에서 끝낸다. GitHub는 최종 커밋과 인계에만 사용한다.}}
\]

기본 연구 흐름:

```text
LOCAL_RESEARCH
-> LOCAL_VALIDATION_PASS
-> COMMIT_PUSHED
-> REMOTE_COMMIT_VERIFIED
```

### `LOCAL_RESEARCH`

로컬/샌드박스에서 수식, 정확 산술, 비용식, tensor/circuit/program search, prototype, 반례, unit/property/regression test, 가능한 공개 checkpoint 실행, raw evidence와 checksum 생성을 끝낸다.

### `LOCAL_VALIDATION_PASS`

GitHub로 보내기 전에 가능한 범위에서 exact token/output/state, causal information availability, arithmetic/traffic/cache/workspace/fallback 비용, 8 GiB/target-scale equation, strongest negative control, deterministic regeneration과 checksum을 검증한다. 수행하지 못한 것은 `NOT TESTED`다.

### `COMMIT_PUSHED`

로컬 검증이 끝난 연구는 GitHub에서 다시 실행하지 않는다. source/config/result/log/checksum/scientific decision/canonical ledgers/README를 commit·push한다.

### `REMOTE_COMMIT_VERIFIED`

원격 branch HEAD와 commit SHA를 다시 읽어 저장 여부만 확인한다.

기본값:

```text
GITHUB_REEXECUTION_REQUIRED=false
GITHUB_ACTIONS_REQUIRED=false
```

GitHub Actions는 사용자가 명시적으로 요청한 경우에만 예외적으로 사용한다.

## Core resource objective

최소 accounting frame:

\[
T_{\rm token}\ge\max\left(\frac{S_c}{BA},\;r\frac{N}{A}\frac{2P}{F}\right).
\]

credible core는 동시에

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
5. `docs/research/RESEARCH_PRIORITIZATION_CONTRACT.md`;
6. `README.md`;
7. conversation memory.

Conversation memory는 authoritative research state가 아니다.

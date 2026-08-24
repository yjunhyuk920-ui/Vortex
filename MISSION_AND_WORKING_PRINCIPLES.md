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

새 EXP 번호를 만들거나 코드를 구현하기 전에 반드시 **기술적 사전판단**을 한다. 아이디어를 만들었다는 이유만으로 실험하지 않는다.

핵심 회차마다 서로 다른 원리 3개를 최소 batch로 만들고, 각 후보에 대해 결과를 보기 전에 다음을 기록한다.

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

연구 우선순위는 다음 직관적 휴리스틱을 사용한다.

\[
\boxed{
\text{Research Value}\propto
\frac{P(\text{works})\times\text{impact if true}}
{\text{cost to falsify}}
}
\]

`P(works)`는 가짜 정밀 확률이 아니라 지금까지의 실패 기록, 자원식, causal information availability, algebraic structure, adversarial counterexample을 반영한 사전 판단이다.

### 구현 전 `NO_GO`

다음 후보는 큰 구현 전에 죽인다.

- 완벽히 성공해도 첫 `10x` core reduction에 못 미친다.
- dominant cost는 그대로인데 보조항만 줄인다.
- 간단한 legal arbitrary-dense 반례가 이미 universal claim을 깨뜨린다.
- 미래 target 정보, 무료 계산, 숨은 compute, 미측정 압축률이 필요하다.
- arbitrary-checkpoint 근거 없이 특정 모델의 low-rank/sparsity/repetition 우연에만 기대한다.
- 이미 닫힌 family의 parameter sweep·rename·nearby variant다.
- 가장 싼 resource equation만으로 8 GiB 또는 p50/p95 목표를 결정적으로 넘는다.

5줄짜리 식으로 죽는 아이디어에 500줄의 코드를 쓰지 않는다.

## `CHEAP_KILL_ONLY`와 `GO`는 종료 상태가 아니다

`CHEAP_KILL_ONLY`는 **즉시 cheapest Gate를 실행하라는 명령**이다.

```text
CHEAP_KILL_ONLY
-> CHEAPEST_KILL 즉시 실행
-> FAIL: 실패 전제 기록 후 새 원리 생성으로 즉시 복귀
-> PASS: 다음 decisive validation으로 이동
```

`GO`는 구현/검증할 가치가 있다는 뜻일 뿐 작동이 확인됐다는 뜻이 아니다.

```text
GO
-> 최소 decisive mechanism 구현
-> frozen local validation 즉시 실행
-> FAIL: 실패 전제 추출 후 새 원리 생성으로 즉시 복귀
-> PASS: 해당 Gate의 VALIDATED_SURVIVOR
```

유망 논문, asymptotic possibility, `CHEAP_KILL_ONLY`, `GO`를 찾았다는 이유만으로 실제 검증을 다음 EXP로 넘기고 현재 회차를 끝내지 않는다.

## 3개는 최소 batch이며 검증까지 닫힌 루프로 반복한다

첫 batch의 세 후보가 전부 `NO_GO`이거나, `GO`/`CHEAP_KILL_ONLY` 후보가 실제 Gate에서 전부 실패하면 회차를 종료하지 않는다.

```text
새 원리 3개
-> PRIOR screen
-> NO_GO 폐기
-> CHEAP_KILL_ONLY 즉시 Gate
-> GO 즉시 최소 validation
-> 실제 생존자 없음
-> 공통 실패 전제 추출
-> 그 전제를 뒤집거나 제거한 새 원리 3개
-> 반복
```

새 batch는 information source, computation order, verification unit, state representation, weight-access dependency, causal schedule, cross-token/cross-layer sharing mechanism 중 최소 하나를 실질적으로 바꿔야 한다. 이름변경, threshold·rank·tile·block sweep, 이미 닫힌 family의 근접변형은 새 원리로 세지 않는다.

전체 폐쇄 루프:

\[
\boxed{
\text{IDEATE}
\rightarrow
\text{PRIOR}
\rightarrow
\text{TEST}
\rightarrow
\begin{cases}
\text{FAIL}\rightarrow\text{IDEATE AGAIN}\\
\text{PASS}\rightarrow\text{NEXT DECISIVE VALIDATION}
\end{cases}
}
\]

## 부분 family 폐쇄는 연구 결과지만 종료 조건이 아니다

한 theorem, finite bound, exact resource equation, adversarial control 또는 exhaustive search가 특정 mechanism family를 결정적으로 닫으면 다음처럼 기록한다.

```text
PARTIAL_FAMILY_CLOSURE
CLOSED_FAMILY=<정확히 닫힌 class>
CLOSURE_EVIDENCE=<proof/result>
OPEN_CLASSES=<아직 합법적으로 열린 class>
ROUND_ACTION=CONTINUE_IDEATION
```

`PARTIAL_FAMILY_CLOSURE`는 영구적인 negative evidence다. 같은 family를 미세변형으로 다시 열지 않는다. 그러나 다른 합법적 실행 원리가 하나라도 남아 있으면 **현재 핵심 연구 회차를 끝낼 수 없다.**

필수 전이:

```text
PARTIAL_FAMILY_CLOSURE
-> 증거 보존
-> 폐쇄 범위 밖의 open classes 명시
-> 공통 실패 전제 추출
-> 새로운 원리 batch 생성
-> 같은 회차에서 연구 계속
```

여러 부분 폐쇄를 누적했다고 해서 자동으로 전체 폐쇄가 되지 않는다. 그 폐쇄 범위들의 합집합이 현재 고정 mission 아래 **모든 남은 legal executor/design class를 실제로 덮는다는 coverage argument**가 있어야 한다.

## 연구 회차를 끝낼 수 있는 유일한 두 상태

### `VALIDATED_SURVIVOR`

최소 한 후보가 현재 rung에서 사전 고정된 decisive Gate를 **실제로 통과**해야 한다.

```text
ROUND_EXIT_REASON=VALIDATED_SURVIVOR
SURVIVOR=<mechanism>
SURVIVOR_GATE=<실제로 통과한 Gate>
SURVIVOR_EVIDENCE=<result/log/checksum>
NEXT_UNTESTED_RUNG=<다음 검증 경계>
```

### `FULL_DESIGN_SPACE_STRUCTURAL_CLOSURE`

이 상태는 현재 고정 mission 아래 **남은 admissible design space 전체**가 theorem, finite lower bound, information argument 또는 exhaustive structural result의 정량적 범위 안에 들어갈 때만 사용할 수 있다.

```text
ROUND_EXIT_REASON=FULL_DESIGN_SPACE_STRUCTURAL_CLOSURE
CLOSED_DESIGN_SPACE=ALL_REMAINING_ADMISSIBLE_DESIGN_SPACE_UNDER_FIXED_MISSION
CLOSURE_EVIDENCE=<proof/result>
OPEN_CLASSES=[]
UNRESOLVED_CLASSES=[]
COVERAGE_ARGUMENT=<모든 남은 legal class가 증명 범위 안에 있다는 논증>
```

아직 열려 있는 합법적 mechanism family, information source, computation order, verification unit, state representation, weight-access dependency, causal schedule, cross-token/cross-layer sharing mechanism을 하나라도 이름 붙일 수 있다면 이 종료 상태는 무효다.

과거의 모호한 `STRUCTURAL_CLOSURE` 라벨은 회차 종료 용도로 사용하지 않는다. 특정 family만 닫혔으면 반드시 `PARTIAL_FAMILY_CLOSURE -> CONTINUE_IDEATION`이다.

다음은 종료 조건이 아니다.

```text
3개 아이디어 실패
유망한 논문 발견
CHEAP_KILL_ONLY 발견
GO 판정
다음 EXP에서 검증 예정
한 mechanism family 폐쇄
여러 family 폐쇄했지만 open class가 남음
PARTIAL_FAMILY_CLOSURE
```

상세 규칙은 [`docs/research/RESEARCH_PRIORITIZATION_CONTRACT.md`](docs/research/RESEARCH_PRIORITIZATION_CONTRACT.md)를 따른다.

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

## 연구·검증·GitHub 운영 원칙

\[
\boxed{\textbf{연구와 검증은 로컬/샌드박스에서 끝낸다. GitHub는 최종 커밋과 인계에만 사용한다.}}
\]

기본 상태 흐름:

```text
LOCAL_RESEARCH
-> LOCAL_VALIDATION_PASS
-> COMMIT_PUSHED
-> REMOTE_COMMIT_VERIFIED
```

로컬/샌드박스에서 수식, 정확 산술, 비용식, tensor/circuit/program search, prototype, 반례, unit/property/regression test, 가능한 public checkpoint 실행, raw evidence와 checksum을 만든다. 수행하지 못한 target 항목은 `NOT TESTED`다.

로컬 검증이 끝난 연구는 GitHub에서 다시 실행하지 않는다. source/config/result/log/checksum/scientific decision/canonical ledgers/README를 commit·push하고 원격 SHA를 다시 읽어 저장 여부만 확인한다.

기본값:

```text
GITHUB_REEXECUTION_REQUIRED=false
GITHUB_ACTIONS_REQUIRED=false
```

GitHub Actions는 사용자가 명시적으로 요청한 경우에만 예외적으로 사용한다.

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
NO_GO
CHEAP_KILL_ONLY
GO
VALIDATED_SURVIVOR
PARTIAL_FAMILY_CLOSURE
FULL_DESIGN_SPACE_STRUCTURAL_CLOSURE
CONTINUE_IDEATION
LOCAL_RESEARCH
LOCAL_VALIDATION_PASS
LOCAL_VALIDATION_FAILED
COMMIT_PUSHED
REMOTE_COMMIT_VERIFIED
BLOCKED_REMOTE_WRITE
NOT_TESTED
```

`STRUCTURAL_CLOSURE` 단독 라벨은 round-exit vocabulary에서 폐기한다.

## Authoritative source order

1. 원격 branch/commit metadata;
2. commit된 raw result/checksum/log;
3. `RESEARCH_STATE.md`, `NEXT_EXPERIMENT.md`, canonical ledgers;
4. 이 문서와 `AGENTS.md`;
5. `docs/research/RESEARCH_PRIORITIZATION_CONTRACT.md`;
6. `README.md`;
7. conversation memory.

Conversation memory는 authoritative research state가 아니다.

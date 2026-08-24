모든 핵심 연구 회차는 기존 기법의 조합·미세변형을 시작점으로 삼지 말고, 현재 병목을 만드는 숨은 전제·계산 순서·정보 흐름·검증 단위 중 최소 하나를 뒤집은 서로 다른 새 실행 원리 3개를 최소 batch로 먼저 발명한다.

그러나 3개를 만들었다는 이유만으로 하나를 숙제처럼 구현하지 않는다. 각 후보는 결과를 보기 전에 다음 사전판단을 통과해야 한다.

```text
PRIOR=<HIGH|MEDIUM|LOW>
WHY_IT_MIGHT_WORK=<구조적 성공 이유>
WHY_IT_MIGHT_FAIL=<가장 강한 반례/하한>
DOMINANT_TERM_CHANGED=<비용식의 어떤 지배항이 사라지거나 크게 줄어드는가>
MAX_IMPACT_IF_TRUE=<성공 시 최대 효과>
ARBITRARY_CHECKPOINT_ARGUMENT=<임의 dense checkpoint에서도 버틸 이유>
CHEAPEST_KILL=<가장 싼 결정적 반증>
FALSIFICATION_COST=<LOW|MEDIUM|HIGH>
IMPLEMENTATION_COST=<LOW|MEDIUM|HIGH>
DECISION=<GO|NO_GO|CHEAP_KILL_ONLY>
```

연구 우선순위는 다음을 직관적 휴리스틱으로 사용한다.

\[
\boxed{
\text{Research Value}\propto
\frac{P(\text{works})\times\text{impact if true}}
{\text{cost to falsify}}
}
\]

`P(works)`는 가짜 정밀 확률이 아니라 지금까지의 실패 기록, 자원식, causal information availability, algebraic structure, adversarial counterexample을 반영한 사전 판단이다.

다음 후보는 구현 전에 `NO_GO`한다.

- 완벽히 성공해도 첫 10× core reduction에 못 미치는 후보;
- dominant cost는 그대로인데 auxiliary constant만 줄이는 후보;
- 간단한 legal arbitrary-dense counterexample가 이미 universal claim을 깨는 후보;
- future target state, perfect selector, 무료 계산, 숨은 compute, 미측정 compression에 의존하는 후보;
- arbitrary-checkpoint 근거 없이 특정 모델의 low-rank/sparsity/repetition에 기대는 후보;
- 이미 닫힌 family의 parameter sweep, rename, nearby variant;
- 가장 싼 resource equation에서 이미 8 GiB 또는 p50/p95를 결정적으로 넘는 후보.

5줄짜리 식이나 작은 adversarial control로 죽는 아이디어에 큰 구현을 하지 않는다.

`LOW` prior라도 성공하면 원본 dense sweep 제거, 한 weight scan으로 여러 exact causal state/token 전진, 또는 \(r\)의 order-of-magnitude 감소처럼 지배식을 바꾸는 효과가 있고 반증 비용이 매우 싸다면 `CHEAP_KILL_ONLY`로 최소 Gate를 먼저 수행할 수 있다.

하지만 `CHEAP_KILL_ONLY`는 종료 상태가 아니라 즉시 실행 명령이다.

```text
CHEAP_KILL_ONLY
-> CHEAPEST_KILL 지금 실행
-> FAIL: 실패 전제 추출 후 새 원리 생성 루프로 즉시 복귀
-> PASS: 다음 decisive validation으로 이동
```

`GO`도 종료 상태가 아니다.

```text
GO
-> 최소 decisive mechanism 구현
-> local validation 지금 실행
-> FAIL: 실패 전제 추출 후 새 원리 생성 루프로 즉시 복귀
-> PASS: VALIDATED_SURVIVOR
```

## 3개는 최소 batch이며 검증까지 닫힌 루프로 반복한다

첫 batch의 세 원리가 전부 `NO_GO`이거나, 검증에 들어간 모든 후보가 실패하면 연구 회차를 종료하지 않는다.

```text
3개 생성
-> prior screen
-> NO_GO 폐기
-> CHEAP_KILL_ONLY 즉시 Gate
-> GO 즉시 최소 validation
-> 실제 생존자 없음
-> 공통 실패 전제 추출
-> 전제를 뒤집거나 제거한 새 3개 생성
-> 반복
```

새 batch는 information source, computation order, verification unit, state representation, weight-access dependency, causal schedule, cross-token/cross-layer sharing mechanism 중 최소 하나를 실질적으로 바꿔야 한다.

Threshold·seed·rank·tile·block 변경, 이름변경, 같은 닫힌 family의 근접변형은 새 원리로 인정하지 않는다.

## 검증 실패 후 복귀가 의무다

다음은 금지한다.

```text
유망 후보 발견
-> 실제 Gate는 다음 EXP로 넘김
-> 현재 연구 종료
```

반드시 다음처럼 진행한다.

```text
유망 후보 발견
-> 현재 회차에서 실제 Gate 실행
-> 실패하면 바로 새 원리 batch 생성
-> 성공하면 다음 decisive validation
```

즉 `CHEAP_KILL_ONLY`나 `GO`를 찾는 것은 연구 회차의 중간 상태일 뿐이다.

## 부분 family 폐쇄는 종료 상태가 아니다

특정 mechanism family가 theorem, finite lower bound, exact resource equation, adversarial control 또는 exhaustive search로 결정적으로 닫혀도 다른 legal class가 남아 있으면 다음 상태로 기록한다.

```text
PARTIAL_FAMILY_CLOSURE
CLOSED_FAMILY=<정확히 닫힌 class>
CLOSURE_EVIDENCE=<proof/result>
OPEN_CLASSES=<아직 열린 class>
ROUND_ACTION=CONTINUE_IDEATION
```

그 폐쇄는 영구 negative evidence로 보존하지만 회차는 계속한다.

```text
PARTIAL_FAMILY_CLOSURE
-> 증거 보존
-> 폐쇄 범위 밖 open classes 명시
-> 공통 실패 전제 추출
-> 새 원리 batch 생성
-> same-round PRIOR/TEST 계속
```

여러 부분 family를 닫았다는 사실만으로 전체 폐쇄라고 부르지 않는다. 모든 남은 admissible class를 실제 증명 범위가 덮는 coverage argument가 필요하다.

## 연구 회차 종료 조건

핵심 연구 회차는 다음 둘 중 하나에서만 끝낼 수 있다.

```text
VALIDATED_SURVIVOR
FULL_DESIGN_SPACE_STRUCTURAL_CLOSURE
```

`VALIDATED_SURVIVOR`는 최소 한 후보가 현재 rung의 frozen decisive Gate를 실제로 통과한 경우다.

`FULL_DESIGN_SPACE_STRUCTURAL_CLOSURE`는 현재 고정 mission 아래 남은 admissible design space 전체를 닫는 theorem, finite lower bound, information argument 또는 exhaustive structural result가 있는 경우다.

전체 폐쇄를 주장하려면 최소 다음이 모두 필요하다.

```text
CLOSED_DESIGN_SPACE=ALL_REMAINING_ADMISSIBLE_DESIGN_SPACE_UNDER_FIXED_MISSION
OPEN_CLASSES=[]
UNRESOLVED_CLASSES=[]
COVERAGE_ARGUMENT=<모든 남은 legal class가 증명 범위 안에 있다는 논증>
```

열려 있는 legal mechanism family를 하나라도 이름 붙일 수 있으면 전체 폐쇄가 아니다. 과거의 모호한 `STRUCTURAL_CLOSURE` 단독 라벨은 회차 종료 용도로 사용하지 않는다.

다음은 종료 조건이 아니다.

```text
3개 아이디어 모두 실패
유망 논문 발견
CHEAP_KILL_ONLY 발견
GO 판정
다음 EXP에서 검증 예정
한 mechanism family 폐쇄
여러 mechanism family 폐쇄했지만 open class가 남음
PARTIAL_FAMILY_CLOSURE
```

상세 규칙: `docs/research/RESEARCH_PRIORITIZATION_CONTRACT.md`.

연구 운영은 `로컬 연구·로컬 검증·GitHub 커밋 인계`로 고정한다. 수식·탐색·프로토타입·반례·단위시험·public checkpoint 실행·raw evidence·checksum 생성까지 로컬/샌드박스에서 완료한다. 로컬 검증이 끝난 회차는 GitHub에서 같은 실험을 다시 돌리지 않는다. GitHub는 source/config/result/log/checksum/docs/README를 commit·push하고 원격 SHA를 읽어 다른 세션으로 인계하는 용도로만 사용한다.

기본 상태 흐름:

```text
LOCAL_RESEARCH
-> LOCAL_VALIDATION_PASS
-> COMMIT_PUSHED
-> REMOTE_COMMIT_VERIFIED
```

GitHub Actions, clean-room reproduction, hosted checkpoint rerun은 기본 요구사항이 아니다. 사용자가 명시적으로 요청한 경우에만 예외적으로 실행한다.

매 의미 있는 repository 회차에서 `README.md`의 목표, 작업 방식, 활성 frontier, 최신 권위 결과, quick start, repository map, advertised capability/test count가 현재 사실과 일치하는지 확인한다. 바뀐 내용이 있으면 같은 commit에 README를 갱신한다.

## Reality-first authoritative execution

Every new core Gate has one authoritative arm: `REAL_EXECUTOR_ONLY`.
Future target tokens or hidden states, perfect selectors, free `N/A`, free transforms, free metadata/workspace, free repair/fallback, unmeasured compression, and peak throughput presented as sustained throughput are forbidden from satisfying a promotion threshold. Synthetic or target-seeing calculations may appear only as non-authoritative debugging diagnostics.

The authoritative arm must execute a finite-word causal path and charge candidate generation, every target position, verification, mismatch repair, rollback, fallback, transforms, packing, metadata, storage/host/device bytes, KV/cache, workspace, fragmentation, and measured wall time where available. Missing target quantities remain `NOT TESTED`; they are never replaced by an ideal grant.

# VORTEX fixed mission and working principles

This file is the canonical compact answer to **“What is our goal and what are our working principles?”**  
Every AI or human session must read it immediately after `AGENTS.md`. It is normative, not an informal summary.

## Canonical answer — 한국어

### 고정 목표

\[
\boxed{
\text{임의의 공개·미수정 Hugging Face Dense 405B급 모델을}
}
\]

\[
\boxed{
\text{재학습·미세조정·증류·가중치 수정 없이 실행기만 교체하여}
}
\]

\[
\boxed{
\text{단일 8 GiB GPU에서 원본 출력 및 successor-state 계약을 보존하면서}
}
\]

\[
\boxed{
\text{같은 머신의 native 4B Q4에 가까운 지연으로 실행한다.}
}
\]

최종 성능 성공선:

\[
p50 \le 1.2\times T_{\mathrm{native\ 4B\ Q4}},
\qquad
p95 \le 1.5\times T_{\mathrm{native\ 4B\ Q4}}.
\]

### 절대 원칙

- 임의의 공개 Dense checkpoint를 대상으로 한다. 특정 prompt, row, layer, checkpoint의 우연을 보편 해법으로 승격하지 않는다.
- retraining, fine-tuning, distillation, LoRA, 모델별 사용자 작성 adapter, 의미를 바꾸는 pruning·근사 가중치 수정으로 목표를 회피하지 않는다.
- 선언된 ABI/RNG 조건에서 token뿐 아니라 필요한 successor state까지 원본 계약을 보존한다.
- 실패 시 원본 405B 전체 재실행이 필요하면 fallback 비용을 숨기지 않고 전부 계상한다.
- GPU VRAM, CPU RAM, SSD, PCIe, HBM, KV, sidecar, metadata, decompression, packing, verification, repair, synchronization, candidate 수 \(N\), commit token 수 \(A\)를 모두 계상한다.
- 원격 또는 숨은 대형 연산을 사용하고 단일 8 GiB 실행이라고 부르지 않는다.
- `MEASURED`, `DERIVED`, `PROJECTED`, `UNVERIFIED`를 분리하고, 측정하지 않은 값은 `NOT TESTED`로 남긴다.
- 결정적으로 탈락한 방법을 threshold, seed, rank, tile, block length만 바꾸어 반복하지 않는다.
- 핵심 연구 회차마다 숨은 전제·계산 순서·정보 흐름·검증 단위 중 하나 이상을 뒤집는 서로 다른 새 원리 3개를 먼저 만든다.
- 그중 원본 dense arithmetic 또는 weight movement를 최소 10× 줄일 수 있는 명시적 수식 경로와 가장 싼 반증 Gate가 있는 후보만 구현한다.
- 연구 결과는 채팅이나 임시 파일만으로 완료되지 않는다. 살아남은 후보와 의미 있는 음성 결과는 코드·config·raw evidence·checksums·문서와 함께 원격 commit으로 남긴다.
- README는 선택 사항이 아니다. 목표, 작업 방식, 활성 frontier, 최신 권위 결과, 실행법, 저장소 구조 중 하나가 바뀌면 같은 회차에 갱신한다.

### 계산 운영 원칙

\[
\boxed{
\textbf{연산과 빠른 반복은 샌드박스에서 먼저,}
\quad
\textbf{영속 증거와 최종 재현은 GitHub에서 마지막에}
}
\]

1. `SANDBOX_RESEARCH` — 수식, 새 원리 3개, 비용식, 프로토타입, 반례, unit/property test를 즉시 반복한다.
2. `SANDBOX_GATE` — exactness, \(r\le0.1\) 경로, 8 GiB, I/O/compute roofline을 가장 싼 Gate로 검사한다.
3. 탈락한 근접 변형은 Actions를 반복 실행하지 않는다. 의미 있는 결론은 다음 원격 연구 기록에 보존한다.
4. 생존 후보만 `SOURCE_COMMIT_PUSHED`로 고정한다.
5. GitHub Actions는 독립 재현, 고정 dependency, 장시간 hosted run, artifact/checksum 보존에 사용한다.
6. 결과 commit과 원격 SHA를 읽어 확인한 뒤에만 `REMOTE_COMMIT_VERIFIED`라고 한다.

GitHub Actions가 연구를 “생각하는 장소”가 되어서는 안 된다. 매 탐색 반복을 commit/workflow로 바꾸지 않는다.

---

## Fixed mission — English

Build a universal executor-only runtime for an arbitrary publicly released, unmodified Hugging Face dense Transformer, including 405B-class models, with:

- peak GPU VRAM at or below 8 GiB;
- no retraining, fine-tuning, distillation, LoRA, semantic weight modification, or user-authored model-specific adapter;
- the declared original output and successor-state contract preserved;
- warm p50 time/token at or below `1.2x` and p95 at or below `1.5x` a native 4B Q4 baseline on the same machine;
- independently reproducible evidence from pinned code and checkpoint hashes.

The target may not be silently reduced.

## Core resource objective

Use the dual roofline as a minimum accounting frame:

\[
T_{\rm token}
\ge
\max\left(
\frac{S_c}{B A},
\;
r\frac{N}{A}\frac{2P}{F}
\right),
\]

where:

- \(S_c\): exact losslessly represented checkpoint bytes streamed per target cycle;
- \(B\): effective storage/host/device bandwidth;
- \(A\): tokens actually committed per cycle;
- \(N\): candidate states actually evaluated;
- \(r\): remaining fraction of original dense target arithmetic;
- \(P\): target parameter population;
- \(F\): effective compute throughput.

A credible core path must make all three move together:

\[
A\gg1,\qquad \frac{N}{A}\rightarrow1,\qquad r\ll1.
\]

Accepted length alone is not a core result when \(r=1\).

## Sandbox-first, GitHub-final workflow

### Stage 1 — `SANDBOX_RESEARCH`

Use the available sandbox first for work that does not require target hardware:

- derivations and exact arithmetic;
- combinatorial or tensor searches;
- cost and roofline calculators;
- randomized, exhaustive, and adversarial controls;
- prototype code and focused tests;
- small public-checkpoint runs when the checkpoint is locally available;
- rapid repair of code and experiment design.

Do not wait for GitHub Actions merely to run calculations the sandbox can perform.

### Stage 2 — `SANDBOX_GATE`

Before opening or extending a research branch, establish:

- the precise original operation skipped or replaced;
- causal information availability;
- exactness and fail-closed behavior;
- a credible optimistic route to at least 10× reduction;
- complete resource equations;
- the cheapest decisive negative control;
- the reason the mechanism differs materially from a closed family.

### Stage 3 — `SOURCE_COMMIT_PUSHED`

Only a survivor, a decisive reusable rejection, or meaningful research infrastructure is promoted to the remote repository. Freeze:

- source and dependency versions;
- config and thresholds before results;
- reference implementation and tests;
- claim boundary and stop rule;
- expected evidence layout.

### Stage 4 — `WORKFLOW_RUNNING`

Use hosted workflows only when they add independent value: clean-room reproduction, a pinned public checkpoint, a long deterministic run, or immutable artifact production. Report the branch, source SHA, workflow ID, and current state instead of remaining silent.

### Stage 5 — `RESULT_COMMIT_PUSHED`

Commit raw evidence, processed result, logs, checksums, provenance labels, scientific decision, and affected canonical ledgers.

### Stage 6 — `REMOTE_COMMIT_VERIFIED`

Read back the remote branch head and commit metadata. A local sandbox result, local commit, patch, or workflow artifact without a remote result commit is not a completed round.

## README freshness contract

`README.md` is the public orientation and session-entry document. It must be checked during every meaningful repository round.

Update it in the same round when any of these changes:

- fixed mission or acceptance criteria;
- mandatory workflow or governance;
- latest authoritative completed Gate;
- active research frontier or PR;
- supported command, dependency, quick start, or repository layout;
- a previously advertised capability or test count becomes stale.

Do not preserve a numeric test count unless it is regenerated by a current canonical validation. Do not describe an old experiment as the active barrier.

When no README change is required, the final report must say:

```text
README_CURRENT=true
README_UNCHANGED_REASON=<specific reason>
```

When it is changed:

```text
README_CURRENT=true
README_UPDATED=<paths/sections>
```

## Status vocabulary

Use these states exactly when applicable:

```text
SANDBOX_RESEARCH
SANDBOX_GATE
SOURCE_COMMIT_PUSHED
WORKFLOW_RUNNING
WORKFLOW_FAILED
RESULT_COMMIT_PUSHED
REMOTE_COMMIT_VERIFIED
BLOCKED_REMOTE_WRITE
NOT_TESTED
```

A long external workflow must not make the session look abandoned: report `WORKFLOW_RUNNING` with identifiers, then continue all independent sandbox work.

## Authoritative source order

For current truth, use:

1. remote branch and commit metadata;
2. raw result and checksums;
3. active PR, workflow, and logs;
4. `RESEARCH_STATE.md`, `NEXT_EXPERIMENT.md`, and the canonical ledgers;
5. this mission file and `AGENTS.md`;
6. `README.md` as the synchronized orientation summary;
7. conversation memory last.

Conversation memory is never the authoritative research state.

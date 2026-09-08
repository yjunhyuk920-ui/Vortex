# 암시적 비선형 직접 질의 표현 — 2026-09-09

## 판정

```text
P1 임의 GF(2) query-side image-frame 생산기          구성 성공 / 코어 탈락
P2 exact value/state transition coalescing          구성 성공 / 범용 90% 게이트 탈락
P3 elementwise-safe encoded-state gauge             구조 정리로 탈락
일반 implicit nonlinear finite-word direct query   OPEN
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
```

고정 VORTEX 목표와 CTC는 바꾸지 않았다. 세 원리는 결과를 보기 전에
`PREREGISTRATION.md`에 동결했고, 그 preregistration은
`dab57747b690ecc4ba0562f3af39e781af01a836`로 먼저 원격에 기록됐다.

현재 정본 결과는 다음이다.

```text
results/e0_implicit_direct_query_gate_v3/summary.json
SHA-256 78644fd5ef4b3e0131bf78867e8986718ed02faeaf37b4228fd1c37909a7af29
focused current suite: 15/15 PASS
```

이 회차는 임의 native 405B 실행기를 만들지 못했다. 대신 **임의 GF(2)
행렬에 대해 현재 질의가 직접 주소를 만드는 유한 생산기**를 실제로
구성했고, 나머지 두 방향의 숨은 무료 연산을 더 강하게 닫았다.

## 1. P1 — 질의 자체가 주소가 되는 complete image frame

임의 이진 `m x n` 행렬 `W`의 입력 좌표를 작은 block `B`로 나눈 뒤,
각 block의 모든 0이 아닌 패턴 `p`에 대해 미리

```text
Y[B,p] = W_B p
```

를 저장한다. 런타임에는 현재 right factor `v`의 각 block bit가 곧
저장된 image의 주소가 된다. 0 block은 읽지 않고, 읽은 image들을 XOR하면

```text
Wv = XOR_B Y[B,v_B]
```

가 정확히 나온다. checkpoint별 instruction stream이나 selector scan이 없다.
컴파일러, 주소 규칙, decoder 모두 유한하다.

10/11-bit 등록 partition을 883개 비-embedding 행렬 전체에 적용하면:

```text
원본 이진 source                  403,747,897,344 bits
변환 image 저장                    41,384,749,318,656 bits
변환 저장량                         4,817.8189106583595 GiB
저장/source 비율                    102.50146091385213x

최악 query payload                  40,365,964,800 bits
query payload                        4.699216783046722 GiB
query/source                         0.09997814246350742
등록 8/675 선 대비                   8.435655770358439x
```

즉 **약 90% 원본 이진 read 제거 자체는 실제 구성으로 달성**했다. 하지만
4.82 TiB에 가까운 변환 저장과 8/675 선의 8.44배 query payload 때문에 목표
코어가 아니다. native Q4/BF16/FP32 ordered arithmetic도 아직 해결되지 않았다.

단순히 block을 키우는 것도 안 된다. `8/675`를 맞추는 첫 동일 block 폭은
`b=85`지만 저장/source 비율이 약 `4.55e23x`로 폭증한다.

더 일반적으로, `W a_j` 형태의 linear atom image를 `S`개 저장하고 모든
right factor를 최대 `t`개 atom XOR로 표현하는 **one-sided complete linear
image frame**은

```text
sum_{j<=t} C(S,j) <= (3S/t)^t
```

때문에 모든 `2^n` right factor를 덮으려면 큰 `S`가 필요하다. `n=16384`에서
90% radius만 해도 최소 560,052 atoms, storage/source 34.18x가 필요하며
`8/675` radius에서는 storage/source가 약 `1.05e23x` 이상이다.

이 하한의 범위는 **ONE_SIDED_LINEAR_COMPLETE_IMAGE_FRAME_ONLY**다. 일반
비선형 cell-probe 구조를 닫았다고 말하지 않는다.

## 2. P2 — exact weight-value / accumulator-state 역색인 실행기

각 입력 column `j`에서 동일한 정확한 finite-word weight descriptor를 가진
row들을 묶는다. 현재 `x[j]`에 대해 leaf product를 한 번만 계산하고, 같은
leaf word를 모든 member row의 원래 reduction slot에 넣는다. column 순서와
row별 add 순서는 원본과 동일하므로 선언한 deterministic finite-word ABI에서
bit-exact하다.

하지만 `16384 x 16384`에서 column당 weight가 하나뿐인 최상의 경우에도:

```text
baseline mult        268,435,456
baseline add         268,435,456
candidate mult            16,384
candidate add        268,435,456
candidate fraction   16385/32768 = 50.0030518%
```

이라서 add 자체가 90% 제거 게이트를 막는다. weight가 모두 다르면 100%로
복귀하고 membership visit도 `mn` 그대로다.

마지막 강화는 정적 weight class보다 더 강한 런타임 coalescing을 허용했다.
현재 `(accumulator word, weight word)`가 같은 row transition은 한 번만 계산할
수 있게 했다. 그럼에도 다음 명시적 FP32-exact adversary가 있다.

```text
첫 column weight: 1,2,...,m
나머지 column:    모두 1
query:             모두 1
m=n=16,384
```

누적 정수는 모두 `2^24` 미만이라 IEEE FP32에서 정확하다. 첫 column 뒤 모든
row accumulator가 서로 다르고 이후 공통 `+1`이 그 차이를 그대로 유지한다.
따라서 매 column마다 `(accumulator,weight)` state가 `m`개 전부 다르며:

```text
state/weight pair updates = 268,435,456
baseline updates          = 268,435,456
fraction                  = 1.0
```

이다. 이는 duplicate-value 및 동일 transition sharing 계열을 닫는 것이지,
모든 비선형 state machine을 닫는 정리는 아니다.

## 3. P3 — encoded-state safe gauge

상태를 `z=P h`로 바꾸고 dense map을 `P_out W P_in^-1`로 conjugate하는
방향을 검사했다. 임의 native finite-word에서 확실히 안전한 것은 coordinate
permutation이다. 그러나 row/column permutation은 dense support 개수를 줄이지
못한다.

더 강하게, 이진 coordinatewise AND를 그대로 싸게 유지하는 **임의 비선형
bijection**도 coordinate permutation밖에 없다.

```text
T(x AND y) = T(x) AND T(y)
```

이면 AND가 Boolean lattice의 meet이므로 `T`는 order automorphism이고,
minimal nonzero atom들을 보존한다. Boolean lattice atom은 coordinate basis
vector뿐이므로 `T`는 basis를 순열하고, 전체 vector도 그 순열로 결정된다.

모든 bijection을 `n<=3`에서 exhaustive check한 결과:

```text
n=1: 2 checked,      1 preserving, 0 non-coordinate
n=2: 24 checked,     2 preserving, 0 non-coordinate
n=3: 40,320 checked, 6 preserving, 0 non-coordinate
```

또 입력 두 개와 출력에 서로 다른 invertible linear gauge `A,B,C`를 허용해

```text
C(x AND y) = A(x) AND B(y)
```

를 요구해도, basis cross-term의 support disjointness와 invertibility 때문에
`A=B=C`인 동일 coordinate permutation만 남는다. `n=2`에서는 216개 triple을
전부 검사했고 survivor는 두 coordinate permutation뿐이었다.

따라서 arbitrary dense map을 sparse하게 만들면서 원래 coordinatewise
Hadamard를 그대로 무료로 유지하는 safe gauge는 없다. 단, transformed
nonlinear operator를 **실제로 구성하고 비용까지 지불하는** 더 일반적인
encoded-state 방법은 아직 OPEN이다.

## 4. O1–O6

```text
O1 OPEN
  임의 native checkpoint에서 목표 비용을 만족하는 compiler가 없음.

O2 OPEN
  살아남은 producer로 load/prefill/decode 전체 causal program이 없음.

O3 OPEN
  arbitrary Q4/BF16/FP32 logits/KV/RNG successor의 귀납적 bit-exact 증명이 없음.

O4 OPEN
  세 후보는 비용이 명시됐지만 모두 adverse. 살아남은 whole-model upper bound 없음.

O5 OPEN
  405B/CUDA/<=8GiB/PCIe/SSD/HBM/native4BQ4 p50/p95/TTFT NOT TESTED.

O6 PARTIAL_E0_REPRODUCIBLE
  preregistration/code/tests/v1-v3 JSON/checksum/report는 재검증 가능하지만 전체 이론은 아님.
```

## 5. 다음 구성적 조건

다음 후보는 아래 세 실패를 이름만 바꿔 반복하면 안 된다.

```text
one-sided complete linear image dictionary가 아니어야 함
weight/state duplicate transition sharing이 아니어야 함
product-preserving coordinate gauge가 아니어야 함
```

필요한 것은 여전히 다음 연결을 실제 유한 알고리즘으로 만드는 것이다.

```text
arbitrary unchanged checkpoint
-> finite compiler
-> compact/nonlocal nonlinear representation
-> current causal query/state
-> subdense paid address/read/decode
-> exact native ordered dense effect
-> exact logits/KV/RNG successor
```

selector, decoder, transformed nonlinear operator, expanded cold sidecar, repair,
fallback 중 어느 것도 무료 primitive로 둘 수 없다.

# 암시적 프로그램 운반체 frontier — 2026-09-09

## 판정

```text
P1 address-only alias router              정확한 GF(2) 구성 / 명시적 carrier 탈락
P2 query-time Patricia synthesizer        정확한 GF(2) 구성 / program traffic 탈락
P3 rank-normal encoded state + paid AND   정확한 GF(2) 구성 / dense work 복귀

GENERAL NONLINEAR IMPLICIT PROGRAM SOURCE OPEN
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
```

세 원리는 결과를 보기 전에
`52facca89659224b8a101f75601ec275740c3e2f`로 preregistration을 원격에
고정했다. 현재 정본은:

```text
results/e0_implicit_program_carrier_gate/summary.json
SHA-256 f6f911fce9383ab6b82a1ac4ad5ef79f354a8ca58d79d21ff71193000c8d5070
focused suite 11/11 PASS
```

이다.

## P1 — checkpoint bit를 값이 아니라 주소 라우팅에 넣기

현재 query block `v_B`로 checkpoint와 무관한 표를 만든다.

```text
Q[B,p] = <p,v_B> mod 2
```

그 뒤 compiler가 각 `(row,block)` logical alias를

```text
A[i,B] -> Q[B,W[i,B]]
```

로 연결한다. 런타임은 `W[i,B]`를 일반 값으로 읽지 않고 고정 logical alias만
순서대로 읽는다. 이 구조는 실제로 정확하다.

16,384 square에서:

```text
query table ops + row XOR     28,516,762
원래 leaf+add slots           536,870,912
fraction                     5.3116608411%
```

이라 계산 이벤트만 보면 90% 이상 줄었다.

하지만 checkpoint 정보는 alias relation에 그대로 남아 있다. 모든 alias target을
이어 붙이면 W 전체가 복구되므로 arbitrary W의 lossless route는 최소 `mn` bit를
구별해야 한다. 현재 구현처럼 alias 하나당 64-bit descriptor를 쓰면 883개 등록
행렬 전체에서:

```text
alias count                  40,365,964,800
descriptor                   300.749874 GiB
binary source 대비          6.398601x
8 GiB 상주                  불가
```

이다. TLB/page table/routing hardware가 안 읽는다고 주장하려면 그 persistent
state와 cache miss/fill 비용을 실제로 계산해야 한다. 이번 결과는 이 64-bit
descriptor carrier를 탈락시키는 것이지 모든 nonlinear address encoding의 하한이 아니다.

## P2 — row pattern을 Patricia trie로 query-time 합성

64-column block마다 실제 row pattern만 compressed Patricia trie로 만든다.
query 시 각 edge mask `M_e`에 대해:

```text
parity_child = parity_parent XOR parity(M_e AND v_B)
```

를 한 번 계산하고 leaf의 모든 같은 row에 결과를 전달한다. complete `2^64`
table은 없다. small control에서 모든 query가 direct MatVec와 정확히 일치한다.

16,384 square의 favorable event 수는:

```text
edge evaluations             8,388,096
leaf contributions           4,194,304
event fraction               2.3436546326%
```

이다. 하지만 edge 하나당 checkpoint mask 64-bit word 하나를 실제로 읽으면:

```text
edge-label bits              536,838,144
raw binary matrix 대비       1.99987793x
```

이고 topology/row membership는 아직 더해지지도 않았다. 즉 연산 공유는 생겼지만
arbitrary checkpoint program traffic가 사라지지 않았다. packed trie가 이보다
낫다고 주장하려면 arbitrary worst-case size와 decoder까지 새로 구성해야 한다.

## P3 — safe gauge 제한을 풀고 transformed nonlinearity까지 실제 계산

임의 GF(2) `m x n` W에 대해 elimination으로:

```text
A W B = J_r
```

를 만든다. 입력/출력을 `B^-1`, `A` 좌표로 들고 있으면 dense linear map은
rank-normal form `J_r` 하나가 되어 full-rank square에서는 `n^2 -> n`, 즉
`1/n`까지 줄어든다. arbitrary rectangular small controls에서도 정확성을 검증했다.

이번에는 Hadamard/AND가 transformed coordinates에서도 싸다고 가정하지 않았다.
두 독립 dense map이 AND에서 만나는 최소 graph:

```text
a=W1 x
b=W2 x
c=a AND b
```

에서 projection을 완전히 없애는 좌표를 쓰면 `z1=z2=x`가 되지만, exact AND는:

```text
c=(W1 z1) AND (W2 z2)
```

가 되어 제거했던 두 dense map을 그대로 다시 계산해야 한다.

16,384 control:

```text
isolated linear fraction            1/16384
whole literal transformed graph     1.0000305171x baseline
```

이다. 이는 per-operator rank-normal + literal transformed nonlinearity를 탈락시키는
결과다. 모든 joint graph encoding 불가능을 뜻하지 않는다.

병행 원격 연구 `experiments/codex_gauge_transport_20260909/REPORT.md`는 이 범위를
더 좁혔다. opaque-word butterfly로

```text
E(native_mul(E^-1 z,E^-1 w))
```

를 `O(n log n)` word-XOR + `n` 원래 native product로 정확히 구현했다. 따라서
“모든 non-coordinate transformed gate는 quadratic”이라는 확대 해석은 금지된다.
다만 이 구성도 arbitrary dense `E F_W E^-1`를 빠르게 만들지는 못하며, dense
projection을 decode-full-execute-encode하면 원래 dense work가 그대로 남는다.

또 native coordinate permutation은 algebraic permutation만으로 충분하지 않고
원래 logical leaf/reduction schedule까지 운반해야 finite-word exactness가 보존된다.
현재 missing object는 cheap gate뿐 아니라 cheap arbitrary dense transformed
projection까지 동시에 만드는 checkpoint-dependent encoding이다.

## O1–O6

```text
O1 OPEN
O2 OPEN
O3 OPEN
O4 OPEN
O5 OPEN
O6 PARTIAL_E0_REPRODUCIBLE

405B/CUDA/<=8GiB/PCIe/SSD/HBM/native4BQ4 p50,p95/TTFT NOT TESTED
```

다음 연구는 checkpoint 정보를 값/주소/edge program/conjugated nonlinear node 중
다른 곳으로 옮겨 놓는 것에 그치면 안 된다. 특히 여러 exact query 사이에서
checkpoint 정보를 **동적으로 공유**하되, 이전에 폐기한 sparse-delta/response cache와
달리 dense legal right-factor 변화에서도 state update 자체가 subdense인 새로운 원리가
필요하다.

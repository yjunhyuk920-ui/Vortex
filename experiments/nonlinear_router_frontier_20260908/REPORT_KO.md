# 비선형 adaptive word router Gate 결과

## 판정

```text
25x108 / 50 words / 2 probes 완전 비선형 adaptive 모델: REJECT
양쪽 크기 <=128인 8,256개 local rectangle: 목표 traffic에서 전부 REJECT
새 첫 미폐쇄 지점: 24x225 또는 25x216 / 99 words / 4 probes
완성 producer: 아직 없음
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
```

이전 연구에서는 저장 cell 자체가 arbitrary nonlinear이고 주소도 앞에서
읽은 값에 따라 바뀔 수 있으면 `25 x 108`, 50개의 64-bit word, 2 probes가
단순 차원 계산상 살아 남았습니다. 이번에는 그 빠진 부분을 직접 닫았습니다.

핵심은 first word가 선형이어야 한다는 가정조차 필요 없다는 점입니다.
`D` bit 원본을 arbitrary `w`-bit cell 함수들로 저장했다고 하겠습니다. 한
주소에서 가장 큰 실제 value fiber를 하나 고르면 원본 집합의 최소 `2^-w`
비율이 남습니다. 이를 마지막 probe 직전까지 반복하면 적어도

```text
2^(D-(t-1)w)
```

개의 원본이 같은 선택 경로에 남습니다. 마지막 `w` bit word 하나로 그
경로의 모든 query를 답해야 하므로 공동 출력 패턴은 최대 `2^w`개입니다.
그 query들의 coefficient span 차원을 `r`이라고 하면 선형 projection의
각 fiber 크기는 정확히 `2^(D-r)`이므로

```text
r <= t*w
```

가 강제됩니다. 주소가 매 단계 값에 따라 바뀌어도 최종 route group은 최대
`S^t`개이므로, 전체 rank-one query 집합은 `S^t`개의 `t*w`차원 이하 선형
부분공간으로 덮여야 합니다.

`25 x 108`, `S=50`, `w=64`, `t=2`를 넣으면 한 route group이 포함할 수
있는 rank-one mask의 정확한 최대값은

```text
2^108 + 2^21 - 3
```

입니다. 2,500개 route를 전부 가장 유리하게 써도 전체 rank-one query의
약 `0.00745058%`밖에 덮지 못합니다. 따라서 **first/second cell 모두
arbitrary nonlinear, second address도 first value에 따라 adaptive, decoder도
arbitrary deterministic logic**이어도 이 2-probe producer는 존재할 수 없습니다.

같은 Gate를 side 128까지 전부 돌린 결과:

```text
전체 rectangle                 8,256
64-bit probe 1개도 예산 밖      2,316
새 cover Gate로 탈락             5,940
남은 shape                         0
```

새로운 첫 미폐쇄 local 지점은 면적 5,400입니다.

```text
24 x 225 : 99 words, 4 probes
25 x 216 : 99 words, 4 probes
4*64/(4*5400) = 8/675
```

하지만 이것은 **가능하다는 뜻이 아닙니다.** 현재 Gate가 더 이상 배제하지
못한다는 뜻뿐입니다. 실제 99-word encoder, 4-probe address generator,
decoder는 아직 없습니다. 다음 핵심 작업은 이 세 가지를 실제로 만들거나,
이 4-probe frontier를 더 강한 정리로 닫는 것입니다.

여기서 한 단계 더 닫았습니다. 4개 word 주소가 query만으로 미리 정해지는
완전 nonadaptive 모델은 coverage 상한이 각각 `0.2243743...`, `0.1121871...`
이라 두 shape 모두 탈락합니다. 첫 word의 **값을 한 번 본 뒤** 나머지 3개
주소를 한꺼번에 정하는 one-value-stage 모델도 각각 `0.8974972...`,
`0.4487486...`이라 탈락합니다.

따라서 지금 남은 4-probe 구조는 단순 nonlinear 저장이 아니라, 두 번째나
세 번째로 읽은 실제 word 값이 **다음 주소를 다시 바꾸는 다단계 adaptive
routing**을 반드시 사용해야 합니다. 이것도 가능성 증명이 아니라 다음
구성의 필요조건입니다.

여기에 32-query 물리 합집합 Gate를 붙였습니다. 한 source에서 전체 rank-one
query들이 실제로 읽는 encoded word 주소의 합집합을 `U(W)`라 하겠습니다.
rank-one query 집합에는 모든 matrix unit이 포함되어 있으므로, `U(W)`의 주소와
그 word 값만 알면 adaptive transcript를 그대로 재생하여 모든 source bit를
유일하게 복원할 수 있습니다. 따라서 모든 source의 `|U(W)|`가 83 이하라면

```text
sum(j=0..83) C(99,j) 2^(64j)
```

개의 descriptor만으로 `2^5400`개의 source를 모두 표현해야 하는데, 정확
계산상 부족합니다. 84개까지 허용해야 처음 capacity가 생깁니다. 즉 어떤
source는 전체 query군을 서비스할 때 최소 84개의 encoded word를 활성화합니다.

독립적으로 rank-one query 32개를 고를 수 있는 강한 E0 interface에서는 그
84개 중 새 word를 하나씩 추가하는 query를 greedy하게 고를 수 있으므로,
어떤 32-query tuple은 최소 32개의 서로 다른 word를 읽게 됩니다.

```text
32 * 64 bit = 2,048 bit
2,048 / (4 * 5,400) = 64/675
등록 target             = 8/675
배수                    = 8x
```

따라서 `24x225`, `25x216`의 남은 multi-stage 4-probe local route도 **독립적으로
선택 가능한 32-query interface에서는 target을 8배 초과하여 탈락**합니다.

중요한 경계가 하나 남습니다. 이것은 실제 batch=1 Transformer의 한 합법적
causal continuation이 그 adversarial 32-query tuple을 반드시 실현한다는
증명은 아닙니다. 따라서 전체 VORTEX 불가능성으로 확대하지 않습니다. 다음
핵심은 이 abstract hard tuple을 합법적 causal state/query에 연결하는 유한
reachability construction 또는, 그 연결을 피하면서도 exact logits/KV/RNG를
생성하는 전역 nonlinear producer입니다.

405B, CUDA, <=8 GiB GPU, PCIe/SSD, native4BQ4 p50/p95, TTFT는 이번 E0
정리에서 실행하지 않았고 그대로 `NOT TESTED`입니다. O1-O5는 OPEN,
O6도 전체 이론 기준으로는 PARTIAL/OPEN입니다.

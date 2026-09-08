# Direct global finite-word producer 연구 결과 — 2026-09-09

이번 회차도 고정 VORTEX 목표를 달성했다고 판정하지 않는다.

```text
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
O1 OPEN
O2 OPEN
O3 OPEN
O4 OPEN
O5 OPEN
O6 PARTIAL
```

핵심 결과는 세 가지다.

첫째, 전역 8 GiB advice로 원본 checkpoint를 복원한 뒤 native dense 연산을
수행하는 P1은 구조적으로 부족하다.  Q4 전체 source에서 advice가 대신할 수
있는 정보는 최대 `4.255098%`뿐이므로 최악 경우 `95.744902%`의 source를
추가로 읽어야 한다. 그 뒤 dense arithmetic도 100% 남는다.

둘째, 수학적 exact sum과 native rounding을 분리하는 P2도 새로운 쉬운
primitive를 만들지 못했다. balanced FP32 RNE에서

```text
[2^24, b, -2^24, 0, -b, 0, 0, 0]
```

의 exact sum은 항상 0이지만 rounded subtree 결과는 정확히 `-b`다. 이를
좌표별로 붙이면 exact sum은 계속 0인 상태에서 rounded 결과가 arbitrary
binary `popcount(W_row & v)`를 담는다. 즉 rounding witness 자체가 direct
MatVec 정보를 다시 계산해야 한다.

셋째, P3에는 새 **global nonlinear route-span theorem**을 만들었다. arbitrary
nonlinear 64-bit cell, cross-matrix mixing, fully adaptive address를 모두 허용해도
한 final route의 모든 simultaneous full-Mv coefficient mask span은 `<=t*64`다.
883개 matrix를 직접 합쳐 계산하면 최소 `578,619` word probes가 필요하지만,
이는 favorable target `299,072,516` words의 `0.193471%`에 불과하다. 따라서
이 theorem은 global advice synergy를 정당하게 계상하지만 목표 불가능성을
증명하기에는 약 516.87배 약하다.

lower bound에서 멈추지 않고 arbitrary GF(2) matrix에 대한 실제 finite
producer도 구현했다. Gauss–Jordan으로 `R=E W`와 row-XOR program을 compile하고,
runtime에서 `Rv`를 계산한 뒤 program을 역순 적용해 `Wv`를 정확히 복구한다.
무작위 rectangular/square controls는 모두 exact였다. 그러나 고정
16,384폭 adversary에서 runtime operation fraction이

```text
16385/32768 = 50.0030518%
```

이고 row-program metadata lower bound만 `447.97 MiB`/matrix라서 >=90% 제거
Gate를 실패한다. 이 실패는 해당 Gauss–Jordan producer에만 적용하며 모든
GF(2) data structure의 불가능성을 뜻하지 않는다.

따라서 다음 핵심은 **원본을 대부분 복원하지 않고, exact-sum witness로
우회하지도 않고, quadratic static program을 읽지도 않는 direct nonlinear
GF(2)/native finite-word global producer**다.

405B 실제 실행, CUDA, <=8 GiB GPU, PCIe/SSD/HBM, native4BQ4 p50/p95, TTFT는
이번 회차에서 실행하지 않았고 계속 `NOT TESTED`다.

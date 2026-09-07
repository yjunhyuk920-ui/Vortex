# Dense interval–residue source: 복원기는 완성, 저비용 정보원은 미확보

2026-09-07. 기준: Vortex PR #140, `58632111980c7b271f0e7bd276fb489560c14a66`.

## 결론과 목표 보존

전체 목표는 임의 공개·미수정 HF dense 405B, 단일 GPU peak 8GiB, 원본 출력/RNG/필요한 다음 상태, 같은 머신 native4B Q4 대비 p50<=1.2x/p95<=1.5x와 기존 TTFT이다. 수정·학습·원격 추론·무료 전처리·무계상 fallback으로 대체하지 않는다.

이번 결과는 **출력의 모든 좌표가 달라도, 정해진 범위와 나머지에서 정확한 정수 출력을 복원하는 구체적 생성기·질의기**이다. 희소한 오차 위치를 찾던 방법과 달리 오차 support 크기를 제한하지 않는다. 그러나 사용한 인증 범위에서는 나머지 표현이 잔차 계수 자체를 정확히 복원할 만큼 커지며, 실제 질의기는 그 계수 비트들을 전부 읽는다. 핵심 엔진으로 채택하지 않았다.

`THEORY_STATUS=NOT_ESTABLISHED`, `CORE_ADMISSION=false`, `HARDWARE_STATUS=NOT_TESTED`, `FULL_MISSION_O1_O6=OPEN`, `THREE_QUALIFYING_NEW_PRINCIPLES=false`.

3방향 비교를 했지만 10배 이상의 전체 경로가 보장되는 새로운 원리 3개를 만들었다고 주장하지 않는다. 아래 코드는 비용 관문 이후의 작은 보조 검증이다. GPU 백엔드나 큰 표를 만들지 않았다. 이 결과로 목표의 실현 가능성이 높아졌다고 판정하지 않는다.

## 1. 목표 정리와 의무

모든 목표 체크포인트 W, 합법 입력·과거·RNG 상태에 대해 유한 자동 생성 C(W)와 실행 E가 원본 관측 및 모든 합법 연속 실행을 보존하며, 준비·저장·상태·읽기·실제 연산·지연이 목표 안에 들어간다는 정리가 최종 요구다. 이번에는 그 정리를 증명하지 못했다.

| 의무 | 이번 범위 | 전체 상태 |
|---|---|---|
| O1 자동 생성 | 제한된 정수 행렬의 유한 부호화기 | HF 전체 미구성 |
| O2 인과 프로그램 | 현재 입력에서 선형 출력; 완벽한 잔차 공급자 없음 | 전체 비선형 본체·상태 전이 미구성 |
| O3 정확성 | 아래 정수/BF16 충분조건의 증명 및 C 비교 | 일반 native 합산·KV·RNG 미검증 |
| O4 전 비용 | 실제 바이트 형식·읽기·popcount·준비 복잡도 | 목표 상한 미확보 |
| O5 규모 예산 | 10배 source 관문 미통과 | 8GiB·지연 미검증 |
| O6 재현 | 사전등록, 입력/출력/원본 BF16/프로그램/테스트/해시 | 한정된 결과만 검증 |

## 2. 세 방향과 가장 싼 검사

A. **희소 오차가 아니라 좁은 값 범위**: 출력 전체에 오차가 있어도 작은 modulus로 각 값을 복원한다. 희망 경로는 싼 예측값+인증 구간+극소수 잔여 비트. 문제는 잔여 비트를 실제로 만드는 계수 정보가 충분히 작아지는가다. 행 L1 경계를 쓰면 아래 계수 복원 정리가 즉시 비용 문제를 드러낸다.

B. **중간 복원을 아예 미루는 합동 상태**: 각 선형층의 큰 값을 복원하지 않고 나머지만 다음 비선형층에 전달한다. 필요한 조건은 합동 상태가 같은 두 값이 다음에도 동등해지는 것. -1과3은 mod4에서 같지만 ReLU 후0과3으로 달라진다. SiLU에서도 부호가 다르다. 따라서 이번의 residue-only 상태는 닫혀 있지 않다. 일반 상태 압축의 불가능성은 아니다.

C. **출력별 residue 대신 공동 code**: 모든 오차를 몇 개의 공동 syndrome으로 판별한다. decoder가 고정된 부가정보와 syndrome만 받는 모형에서 [-1,0,1]^m을 구별하려면 최소 ceil(m log2 3)비트가 필요하다. m=8의6561오차를 4bit code로 부호화하는 실제 표에서는16syndrome만 나오고 충돌한다. 이 모형은 decoder가 x와W의 추가 함수를 계산할 수 있는 일반 알고리즘을 배제하지 않는다.

## 3. 정확한 복원식

정수 y가 인증된 구간 L<=y<=U에 있고 U-L<q라면, r=y mod q로부터

    y = L + ((r-L) mod q)

로 유일하게 복원한다. 결과가U를 넘으면 구간에 맞는 대표가 없는 것이므로 거부한다. 두 가능한 대표는q의 배수만큼 달라지므로 구간에 두 개가 있을 수 없다는 것이 증명이다. 구간 자체가 거짓이거나 residue가 잘못되었음을 이 decoder가 독립적으로 탐지한다는 뜻은 아니다.

예: y가[-5,5]에 있고 y mod16=13이면 y=-3이다. 오차가 네 좌표 모두에 있는[-3,2,-1,4]도 residue[13,2,15,4]와 같은 구간으로 모두 복원한다. 위치를 미리 알려줄 필요가 없다.

원본을 W=P+R, P_ij=a_i로 쓴다. a_i는 각 행의 낮은 중앙값이다. 현재 x의 성분이{-1,0,1}이면

    prediction_i = a_i sum_j x_j
    e_i = sum_j R_ij x_j
    |e_i| <= B_i = sum_j |R_ij|
    q_i = 2^bit_length(2 B_i) > 2 B_i

이다. B_i=0은q_i=1로 처리한다. 나머지를 생성한 후[-B_i,B_i]에서e_i를 복원하고prediction을 더한다. R을 버리지 않는다.

## 4. 나머지 정보원: 실제 저장·주소·연산

실행 파일은12바이트 헤더, 행마다8바이트의a_i/비트폭/B_i, 그리고R의 최소 signed bitplane이다. b_i는 실제 행 잔차를 표현할 최소 2의보수 비트 수이다. 원본 배열 없이 재로드한 Program은 blob만 보유한다.

현재 x에서 양수·음수 위치 비트마스크를 각각 만든다. 각 residual bitplane과마스크의 AND/popcount 차이에 해당 비트의 가중치를 곱하여modq_i로 누적한다. 마지막 부호 bitplane에는음의 가중치를 사용한다. q_i가 커졌다고 같은 sign-extension plane들을 반복 저장하지는 않는다. 이는 완전한 손실 없는 잔차 표현이지 1비트 oracle이 아니다.

직렬화 크기와 한 질의의 읽는 필드량은

    S = 12 + 8m + ceil(n/8) sum_i b_i

이다. 질의에서S를 모두 순회한다. Python int popcount 호출은2 sum b_i이지만 각 operand가n비트이므로 기계 명령 하나처럼 세지 않는다. 64비트 단위 일감의 하한만도2 ceil(n/64) sum b_i이다. 마스크 AND, 누적, shift, modulo, 임시 객체/복사, 구간 lift, 출력/로그, 입력 검사 비용이 추가다. 기록한S는 물리적 HBM/PCIe/SSD 트래픽이나 지연이 아니다.

생성은 모든 W를 읽고, 각 행을 정렬하고, 모든 잔차 비트를 내보낸다. O(mn log n+n sum b_i) 작업, Python 구현은W 객체와blob·행 작업공간을 보유한다. 원본은 별도 보존한다. 405B 스트리밍 loader나 GPU allocation 상한을 구현하지 않았다.

## 5. 이번 방법에서 나타난 결정적인 관계

B_i=||R_i||_1이면 모든 계수는[-B_i,B_i]안에 있다. q_i>2B_i이므로

    R_ij = -B_i + ((R_ij mod q_i + B_i) mod q_i)

이다. **즉 이 방식의 residue 계수는 원래 residual 계수를 일의적으로 복구할 만큼의 정보를 이미 포함한다.** 비트폭도 필요한 signed residual 폭보다 작지 않다.

이는 데이터구조 전체의 읽기 하한이 아니다. 정보가 모두 저장되어 있어도 잘 설계된 주소/공동 계산으로 일부만 읽을 가능성은 남아 있다. 정확한 판정은 **이번 구현은 모든 residual plane을 실제로 순회하며, 손실 없는 표현을 넘는 새 source 절감이 없다**는 것이다.

n=16,384이고각잔차계수가+1/-1인 예에서도L1 경계의B는16,384, 필요 modulus는65,536(16비트)이다. 순진한 residue 행렬은BF16보다 작지 않다. 중복 부호비트를 없앤 ternary 2bit 표현은12.5%이지만, 그것은 기존 저비트 잔차값을 그대로 읽는 경우이고 입력/metadata/연산은 추가다. 이 형식의 수치만으로 8GiB나 속도비를 주장하지 않는다. 다른 codec/입력관계/빠른 residue source는 별도 문제다.

## 6. 정확성 범위와 실행 결과

가중치는BF16으로 정확히 표현되는정수[-127,127], x는정수{-1,0,1}; 행마다양수계수가하나이상있어야 한다. 각행의절댓값합이2^24미만인지 검사한다. 이로써별도FP32곱과균형합산트리의모든중간값이정확한정수이고, 최종BF16 RNE까지보존된다. 양수계수와입력의+0규칙은최종0의부호도+0임을보장한다. -0입력/일반BF16소수/NaN/무한대/CUDA reduction으로확대하지않는다. 범위밖입력은거부하고가중치를양자화하거나실수->정수로강제변환하지않는다.

사전등록한18행렬/288입력에서19,968출력이정확한정수reference와C의native FP32/BF16reference에모두일치했다. 91입력에서는모든출력좌표의보정값이0이아니었다. 그러므로성공이희소오차때문만은아니다. 일반조밀/작은조밀잔차12행렬은mod65521 full-rank 증서, constant6행렬은rank1이다.

128x128 결과:

| 집단 | 원본 BF16 | 실제 source | 원본 대비 | 더공정한원본signed-bitpack payload대비 |
|---|---:|---:|---:|---:|
| dense uniform seed1701 |32768B|16940B|51.70%|118.16%|
| dense uniform seed1702 |32768B|16908B|51.60%|117.94%|
| 작은조밀잔차 두seed |32768B|5132B|15.66%|45.56%,47.17%|
| constant rows 두seed |32768B|1036B|3.16%|9.59%,9.86%|

마지막집단은행마다같은수인rank1 양성대조군이다. 일반 dense 성공으로 세지 않는다. 작은잔차집단도의도적으로그렇게구성한대조군이다. 정수원본자체를7bit정도로저장할수있으므로, dense uniform에서는이번source가오히려커진다.

준비와8질의를포함한최소필드예산은(2mn+S+8S)/(8*2mn)이다. 원본읽기와source기록/질의만세며sort/읽기복사/재로드/상태/연산은추가다. dense128에서는약70.55~70.66%이다. 이역시latency가아니다.

## 7. 원본 수치·상태로 확장할 때의 한계

정확한정수합이나그residue만으로일반native합산을재현할수없다. [2^24,1,-2^24,1]과[2^24,-2^24,1,1]은정확한합2지만명시한FP32균형트리에서1과2다. 원본의각반올림효과를싼정보원으로만드는절차는이번에없다. 이를단순반례검사로해결했다고하지않는다.

비선형residue 상태의반례는ReLU(-1)=0,ReLU(3)=3(mod4)이다. SiLU(-1)/SiLU(3)의BF16부호도다른보조수치를기록했다. 이값은math.exp->FP32->BF16의보조계산이며CUDA/전영역exp증명은아니다. 벡터공동code실험의decoder정보제한도결과JSON에명시했다.

## 8. 재현과 연구 인계

`python run.py`는C를재컴파일하고모든input/program/output/증거를생성한다. `python -m unittest discover -s tests -v`는16검사를수행한다. `python verify.py`는현재manifest를보존한뒤run을다시수행하여111과학파일의동일성을검사한다. 무작위씨드/범위/기준은PREREGISTRATION.json에고정했다.

입력guard에서실수를int로암묵변환할수있던초기코드를strict검사로고쳤다. 초기소스/초기manifest/로그를history와logs에보존했다. 기존원자료와결과는변경되지않았고,scope의추가SiLU수치와18개원본BF16파일을추가했다. 처음부터있었던자료처럼보고하지않는다.

공개모델/HF/전체KV/RNG/GPU/405B/8GiB/4BQ4/TTFT/전체repo테스트/Actions는미실행이다. source/checker/측정값은작은보조결과다. 다음연구는modulus개수나toy크기를늘리는작업이아니라, **본체의비선형·반올림과상태를보존하면서현재입력의구체적인효과를소량의작업으로만드는정보원**을구성해야한다. 좁은interval이나싼residue generator를기정사실로두지않는다.

## 선행연구

- Uchino/Ozaki/Imamura, High-Performance and Power-Efficient Emulation of Matrix Multiplication using INT8 Matrix Engines, arXiv:2508.03984v1 (https://arxiv.org/html/2508.03984v1). CRT기반행렬계산이기존도구임을명시한다. 본연구는이를그대로구현하거나논문의GPU성능을재현하지않았다. 원문은tall-and-skinny/small matrix성능을대상에서제외한다.
- Uchino/Ozaki/Imamura, Error Analysis of Matrix Multiplication Emulation Using Ozaki-II Scheme, arXiv:2602.02549 (https://arxiv.org/abs/2602.02549). 정확한합/수치오차보장과특정실행의비트동등성은구분한다.
- NVIDIA, Floating Point and IEEE754 (https://docs.nvidia.com/cuda/floating-point/index.html). Native계산순서의차이를다루는배경자료이다.

새대수학이나일반LLM가속의최초발명이라고주장하지않는다. 이번의구체적인추가는희소오차가정없는bounded decoder/source/cost연결과그적용경계의실행가능한검사이다.

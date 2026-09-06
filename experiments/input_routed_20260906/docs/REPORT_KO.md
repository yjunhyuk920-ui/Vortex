# 입력이 고른 경로만 읽는 정확 출력 프로그램: 구성 결과와 비용

2026-09-06. 원격 부모 `147dad60b4450d7f32de6e3b24d276b075aed817`
(`yjunhyuk920-ui/Vortex`, PR #128). 사전 등록: `../PREREGISTER.md`.

## 결론

원본 가중치를 질의 객체에 남기지 않고, 현재 입력의 비트가 선택한 실행 그래프의
경로만 읽어 BF16 출력을 만드는 **실제 생성기·직렬화·실행기**를 구성했다.
선택 절차는 입력 비트 검사이며, 이상적인 selector나 부분합 oracle이 아니다.
하지만 작은 등록 사례에서 코드/주소 읽기가 원본 가중치 payload보다 훨씬 컸고,
대부분은 고정된 생성 자원 한도에 도달했다. 이 표현을 핵심 방식으로 채택하지 않는다.

이는 bounded auxiliary E1 연구이며, 완성 이론이나 실제 모델 성능의 진척이 아니다.
전체 `THEORY_STATUS=NOT_ESTABLISHED`, `CORE_ADMISSION=false`,
`HARDWARE_STATUS=NOT_TESTED`, `O1-O6=OPEN`을 유지한다.

## 1. 새로 검사한 의존성

지난 bitplane 소스는 원본 배열 대신 code를 읽었지만 결국 code 전체를 훑었다.
이번에는 계수를 더 잘 압축하는 대신 **최종 출력의 각 비트를 결정하는 함수**를
전처리하고, 입력에 따라 나머지 프로그램의 한쪽만 선택한다.

각 출력 비트 함수 f, 입력 비트 b에 대해

    f = (not b AND f[b=0]) OR (b AND f[b=1])

가 정확하게 성립한다. f0=f1이면 그 검사를 없애고, 같은 (b,f0,f1)은 공유한다.
최종 BF16 반올림도 그래프 내부에 포함하므로 출력에서 사라지는 낮은 비트는
소거될 기회가 있다. 질의는 현재 입력에서 도달한 노드만 방문하고, 여러 출력은
같은 질의에서의 노드 결과를 공유한다. 이전 입력의 유사성/중복을 요구하지 않는다.

이 분해와 reduced BDD 자체는 선행연구다. 새 학술 패러다임이나 학계 최초로
표현하지 않는다. 예전의 '작은 정확 회로가 있다면'이라는 식에서 실제 구성 절차와
주소 선택, 물리적 직렬화 payload까지 전진한 **특정 참조 구현**이다.
완전한 hot 회로로 405B 정보를 8GiB에 압축했다고 하지 않는다. 코드가 cold에 있어도
모든 접근/주소/페이지 비용을 지불해야 하며 이번 실행은 CPU 메모리 buffer였다.

다른 두 흐름(잔여식으로 rounding-cell 식별, 역방향 native 제약 전파)도 사전 비교했다.
둘 모두 싼 residue 생성기/제약 접근·탐색 상한이 없으므로 그 빈칸을 이상적 도구로
두고 backend를 만들지 않았다. 세 제안을 세 개의 적격 신기술로 세지 않는다.

## 2. 구성 알고리즘과 정확성 범위

`src/routed.py`는 상수 곱셈을 shift/add 회로로, 덧셈을 carry 회로로 만든 뒤
이를 hash-consed BDD 연산으로 합성한다. 가능한 입력들을 나열해서 정답표를 만드는
것이 아니다. 최종적으로 부호, leading bit, guard/sticky/ties-to-even 규칙으로
BF16 비트 16개를 직접 만든다. 도달 가능한 노드만 다시 번호를 붙이고 VRC1으로 저장한다.

VRC1: 28-byte header + 입력 비트 좌표당 4 bytes + 출력 비트 root당 4 bytes
+ 노드당 (변수번호, 0분기, 1분기) 12 bytes. 생성된 파일을 다시 읽어 실행한다.
질의 Program에는 원본 가중치, reference 답, 다음 토큰 또는 constructor cache가 없다.

**고정한 제한된 native 연결:** 가중치는 |w|<=127인 정수, 입력은 canonical signed
4-bit 정수 -8..7을 BF16에 정확히 저장한 값이다. 실제 BF16 활성값을 4비트로
변환한 실험이 아니며, 이 제한을 최종 임무에 채택하지 않는다.

행마다 sum |w_j| * 8 < 2^24를 검사한다. 그러면 각 정수 곱과 모든 부분합이 FP32에
정확히 들어간다. FP32의 별도 곱/덧셈을 +0에서 시작한 결과는 정수 합과 같다.
마지막 BF16 RNE는 그래프의 guard/sticky/짝수 tie 회로와 같다. -0을 따로 표현하는
입력, 임의 지수 BF16, overflow/underflow, native reduction의 일반적 반올림은
이 증명의 대상이 아니다. 그 사례들에 이 정리를 확장했다고 하지 않는다.

정확성은 Boolean gate 합성의 귀납법, ripple-carry/2의 보수 계산과 폭 상한,
leading-bit별 BF16 반올림의 경우 분해로 보인다. 같은 자식의 병합/노드 공유는
함수를 바꾸지 않는다. 이는 테스트 개수만으로 내세운 보증과 구분한다.

## 3. 등록 결과

고정 8개 shape/값 범위, 2 seed, 2 변수 순서: 총 32개 생성 시도.
중간 노드 100,000개 또는 apply cache 300,000개 한도를 사전에 정했다.
결과를 보고 한도를 늘리거나 성공한 순서/seed만 선택하지 않았다.

- 생성 완료 6개, 노드 한도로 거부 26개.
- 완료한 프로그램의 1,160개 입력, 4,640개 BF16 출력: 불일치 0.
- n=2는 두 signed 4-bit 입력의 256개 조합 전체, 다른 완료 사례는 고정 68개 입력.
- 독립적인 정수 RNE와 NumPy FP32 순차 실행 후 BF16 반올림을 모두 비교했다.
- 별도 검사에서 symbolic RNE를 모든 13-bit 정수 8,192개에 대조했다.

| 완료 집단 | 원본 가중치 | 실행 그래프 파일 | 질의 code-read 비율의 각 사례 중앙값 |
|---|---:|---:|---:|
| 입력 2 / 출력 4 | 16 B | 10,240~14,320 B | 162.75~195.00배 |
| 입력 4 / 출력 4 | 32 B | 176,604~200,220 B | 238.00~239.875배 |

이는 CPU 실행기에서 실제로 주소를 선택해 읽은 직렬화 필드의 논리적 payload 집계다.
GPU HBM 트랜잭션, SSD 실제 IO, 실행시간 비율이 아니다. 중간 graph가 커져 거부된
26개는 성공한 빠른 query가 아니며, dense fallback으로 성공으로 바꾸지 않았다.
거부는 이 생성기와 자원 한도의 결과이지, 해당 함수의 최소 회로 크기 하한이 아니다.

**소형 비교의 중요한 한계:** root만 64m 바이트이므로 raw BF16 2mn에 대한 고정비는
32/n이다. 등록한 작은 n에서는 root 고정비만으로도 10% 관문을 통과할 수 없다.
따라서 등록문서의 '소형 모든 사례 <=10%' 조건은 target-scale 성패를 결정하는
검사로는 부적절하다. 이 사실을 숨기거나, 작은 배수들을 405B의 속도로 외삽하지 않는다.
관찰한 것은 실제 selector 구현/코드 팽창/생성 한도이며, 일반 알고리즘 불가능성이 아니다.
조건부 원리를 고정 목표의 유망 core로 올릴 근거도 얻지 못했다.

## 4. 무료로 두지 않은 비용

N=저장 노드수, V(x)=질의에서 실제 방문한 노드수, t(x)=검사한 서로 다른 입력비트수:

    program = 28 + 4np + 64m + 12N bytes
    code_reads(x) = 64m + 12V(x) + 4t(x) bytes

원본 2mn bytes는 별도로 보존한다. 실제 constructor는 검증/범위/합성에서 총
3mn개의 계수 방문을 한다. Apply 호출, unique table/cache, 임시 graph 및 compact copy가
추가된다. 저장 노드 상한은 생성 중의 파이썬 객체/딕셔너리/stack/allocator byte 상한이
아니다. `measurements.json`에는 별도의 로컬 시간/프로세스 peak RSS 관찰을 남겼다.

질의에는 2n-byte 입력, 2m-byte 출력, 노드별 주소/분기, sparse memo의 조회·삽입,
재귀 stack도 필요하다. memo를 매 질의 전체 N에 대해 초기화하는 숨은 스캔은 없다.
메모장 하한 5V bytes는 키/값만으로 실제 Python hash overhead를 포함하지 않는다.
64-byte/4096-byte 주소 블록 수는 배치 진단이며 실제 cache miss/IO 측정이 아니다.
파일 검증은 로딩 때 전체 파일을 스캔하므로 무료 TTFT가 아니다.

작은 그래프와 짧은 경로가 **동시에** 구성되어야 이 방법에 성능 근거가 생긴다.
현재는 전체 native ABI, 전처리 peak, target 크기의 N,V 분포와 보장, 전체 state/RNG,
8GiB 총 peak와 실제 baseline 비율의 충분상한을 어느 것도 닫지 못했다.

## 5. 연구 원칙과 다음 결정

같은 BDD의 node cap, seed, variable order를 더 크게 쓸 것만으로 핵심 연구를
재개하지 않는다. 작은 모델 성능 저하를 전 405B 하한으로 확대하지도 않는다.
재개에는 코드 팽창과 읽기 경로를 **함께** 줄이는 새 native 의존성이나 표현,
그리고 작은 사례의 root 고정비가 지배하지 않는 비용 증명이 먼저 필요하다.
B/C 방향에서도 빠른 소스를 기호로 가정하거나 무료로 역전파하지 않는다.

이번에 완성한 것은 제한된 입력 영역의 실제 참조 프로그램이다. O1-O6 전체가
해결되지 않았고, 이번 결과로 목표 달성 가능성이 높아졌다고 주장하지 않는다.
검증/기록 완료와 과학적 목표 완료를 구분한다.

## 6. 재현과 저장 범위

    python src/experiment.py --out results
    python -m unittest discover -s tests -v

15개 단위 검사 통과. 주요 결과 및 실제 직렬화 프로그램 10개 파일을 두 번
독립적으로 재생성하여 byte-identical 확인. wall-clock/RSS는 비결정적 별도 기록이다.
로컬 ZIP은 생성된 6개 VRC1 파일까지 포함한다. 원격에는 소스·테스트·사전등록,
보고서·검증 manifest, 완전한 원시 관찰의 무손실 압축 기록을 저장한다.
VRC1 파일은 source로 재생성하고 manifest의 SHA256으로 확인하도록 한다.
이전 로컬 bitplane/SwiGLU/event ZIP은 이번 결과에 소급 병합하지 않는다.

## 7. 선행연구와 범위 확인 (2026-09-06 조회)

- Shi et al., On Tractable Representations of Binary Neural Networks, KR 2020.
  https://proceedings.kr.org/2020/91/ — BNN 결정함수의 OBDD/SDD 컴파일. BF16 LLM 엔진 성과 아님.
- Bryant & Chen, Verification of Arithmetic Circuits with Binary Moment Diagrams, DAC 1995.
  https://www.cs.cmu.edu/~bryant/pubdir/dac95a.pdf — word-level 표현과 bit BDD 크기의 차이.
- Chen & Bryant, *PHDD: An Efficient Graph Representation for Floating Point Circuit
  Verification, ICCAD 1997. https://www.cs.cmu.edu/~bryant/pubdir/iccad97.pdf
  저자 초록은 floating-point multiplier의 rounding 이전 검증을 구분한다.
  이를 일반 native 소수 연산 전체가 싸게 컴파일된 증거로 사용하지 않았다.

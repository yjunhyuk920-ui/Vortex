# 실제 공개 모델의 native 전역 전이 생성 — 2026-09-08

## 결론과 전체 목표

고정 목표는 바꾸지 않았다. 임의의 공개·미수정 HF dense405B, batch1,
단일GPU 전체peak<=8GiB, 원본 출력/logits/RNG/필요 다음 상태 보존,
같은 머신 native4BQ4 p50<=1.2x/p95<=1.5x와 기존TTFT 및 전 비용 조건이다.

이번에는 실제 원본 SmolLM2-135M 전체를 사용하는 세 가지 유한 생성·실행
절차를 구현했다. 작은 parity 모형이나 합성 모델의 성공을 확대하지 않았다.
그러나 A/C는 모든 주요 행렬 계산을 유지하고, B는 매번 전체 원본 forward와
상태 codec을 수행한다. 세 방식 모두 핵심 비용 관문 미통과다.

```
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
FULL_MISSION_O1_O6=OPEN
THREE_QUALIFYING_NEW_PRINCIPLES=false
```

여기서 HARDWARE_STATUS는 목표405B/8GiB/4B비교를 뜻한다. 작은 실제 모델의
CPU 실행 자체는 수행했다. 정확한 native 전이를 생성했다는 사실을 고속화나
전체 이론 완성으로 보고하지 않는다.

## 1. 원본, 실제 작업, 재현 범위

- 원본: `HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
- `model.safetensors` 269060552B, SHA256
  `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`.
  공식 LFS SHA와 실제 다운로드 SHA를 대조했다. 계수 payload는269030016B다.
- Python3.12, torch2.8.0+cpu, transformers4.55.4, 원본 BF16, CPU1thread,
  deterministic algorithms. 실제 설치목록은 `requirements.observed.txt`.
- `AutoModelForCausalLM.from_pretrained(...,torch_dtype='auto',local_files_only=True)`.
  재학습, 가중치 변환, 사용자 작성 모델 adapter, trust_remote_code 없음.
- 고정 prefix `[1,42,73,11]`, `[1,314,271,18]`, `[2,77,29,3]`;
  seed260908/260909/260910. 자연어 품질 평가가 아닌 token-ID 인과 실행 검사다.
- 원래 HF `.generate`: sampling,temperature1,top_k0,top_p1,max_new_tokens4,
  originalEOS. 각 arm이 자신의 logits, cache, RNG로 다음 토큰을 선택했다.
  reference 미래 토큰, 참조 KV, 정답 logits를 후보 입력으로 주지 않았다.

전체 호출 관측은 logits와 30층K/V, 총61tensor roots다. 값의 비트뿐 아니라
shape/stride/storage_offset/device와 DynamicCache 길이·필드, forward전후 및
최종 RNG 상태를 기록한다. 관찰용 복원과 비교·hash는 검증 비용이며 런타임
가속 계산서에서 몰래 제외한 무료 정보원이 아니다. 지연 벤치마크는 하지 않았다.

## 2. A — native 관계의 유한 생성·정확 소거

생성기는 실제 HF forward를 `make_fx`로 따라가 native tensor 연산 그래프를
만든다. 출력과 모든 필요한 다음 K/V를 roots로 남긴다. 가능한 입력을
열거한 정답표가 아니며, 입력값을 상수로 덮어쓰는 생성기도 아니다.

변환은 제한된 순수 연산의 **동일 연산·동일 인자** 공유와 root에 연결되지
않는 순수 연산의 제거뿐이다. 소수 연산의 재결합·근사·가중치 변경은 없다.
mutable/random native 연산은 거부하며, 알 수 없는 순수 ATen 연산은 실제로
남겨 실행하고 비용을 센다. 외부 출력의 별도 allocation과 그 alias 경로는
공유 대상에서 제외한다.

저장 프로그램은 모든 node, 연산명, 순서, 인자, 원본 parameter/buffer의 이름과
layout을 JSON으로 기록한다. 재로더가 이를 원본 tensor에 결합해 다시 실행한다.
이번 생성물에는 입력/정답을 담은 literal tensor가 없으며, 각 graph는 원본
tensor273개를 참조한다. **원본 계수가 없어지는 것이 아니라 그대로 필요하다.**

정확성의 제한형 증명: 선언한 deterministic native ABI에서 각 node의 입력이
동일하면 출력도 동일하다. 같은 인자를 갖는 순수 함수의 결과 공유는 값을
보존하며, 모든 관측 root의 조상들을 유지하면 위상순서 귀납으로 출력과 다음
상태를 보존한다. 실제 cache 객체 갱신도 동일한 K/V/layout으로 수행한다.
이는 지원한 API·효과·layout guard를 전제로 한다. 모든 HF 모델/API/CUDA 또는
모든 legal continuation에 대한 증명이 아니다.

## 3. B — 실제 상태 표현 변환과 비용

`E`는 BF16 K/V 전체를 bytepayload와 layout schema로 저장하고 `D`는 같은
값·shape·stride·offset으로 복구한다. 이 codec은 실제 파일/프로그램에 정의된다.

`D(E(s))=s`, `s'_next=E(F_W(input,D(s')))`.

이 관계는 정확한 인과 실행을 구성하지만, 이번 구현은 **매번 F_W 전체**를
호출한다. 입력이 바뀔 때마다 실제 decode와 encode도 수행한다. PackedCache의
기본 layers에는 숨겨진 원본 KV tensor를 남기지 않는다. verifier는 필요할 때
codec으로 관측을 복구하며, 그 일을 런타임의 무료 native state로 세지 않는다.

단순 nativeRoPE 역변환으로 더 싼 좌표계를 만들 수 있는지도 별도로 검사했다.
실제 첫 층의 q/k 값에 고정 위치0,1,7,31을 적용했다. 위치1에서768key좌표 중
144개가 역변환 후 달랐다. 예: 원본-3.421875, native역변환-3.40625.
위치7/31은211/263개가 달랐다. 이 검사는 실제 activation에 합성 위치를 준
범위 검사이며 정상 generation trajectory라고 하지 않는다. rawkey를 그대로
보존하거나 다른 statechart를 사용하는 모든 방법을 배제하지 않는다.

## 4. C — 목표 답을 주지 않는 native demand 실행

실제 구현한 domain은 `TOP` 또는 정확한 native singleton이다. logits/다음KV는
모두 TOP에서 시작하며, 원하는 정답을 검사기에 공급하지 않는다. root를 거슬러
필요한 node를 표시하고, 구체적인 인자들이 준비되면 원래 native 함수를 한번
실행하여 singleton을 만든다. 마지막 사용 이후 임시 참조도 제거한다.

출력 제약을 미리 주지 않았으므로, 이 domain에서 `inverse(TOP)=TOP`이고 이번
backward narrowing은0회다. 최종 실행은 필요한 모든 matrixkernel로 돌아온다.
이는 실제로 종료하는 해석기이지만, 작동하지 않는 완벽한 역해석기나 공짜
SAT/SMT 오라클을 만든 것이 아니다. 더 풍부한 relation domain 전체에 대한
불가능성 정리도 아니다.

## 5. 실제 검증 결과와 마스크 분기 정정

처음 fullprefill graph는61roots의 bit/layout이 일치했다. 첫 generationcapture는
공식 maskbuilder의 `padding_mask.all()`을 tracing할 수 없어 중단됐다.
원시 오류와 당시 소스는 `logs/attempt01_source`, `logs/transition_initial.log`,
`results/run`에 보존했다. 수치 오답이나 일반 알고리즘 불가능성으로 분류하지 않았다.

실제 pinned masking_utils를 읽고, batch1/SDPA/비가공DynamicCache/슬라이딩·custom
mask없음/empty-cacheprefill또는single-token decode/전체길이all-one mask라는
조건을 외부에서 검사했다. 이 조건에서는 원래 branch도 None causal mask를
선택한다. 검사 비용을 지불한 뒤에만 내부 trace에 None을 전달한다. padding을
일반화하거나 검사를 우회하지 않았다.

후속 user 지시에 따라 동일 prefix/seed에서 attention_mask를 명시적으로 전부1로,
pad_token_id는 기존HF자동선택과 같은EOS0으로 지정했다. EOS자체는 바꾸지 않았다.
`run_v2` 원본과 `run_v3` 원본의 logits/KV/layout/RNG/sequence도 모두 일치했다.
자세한 변경 범위는 `PROTOCOL_CLARIFICATION.md`에 남겼다.

각 후보 A/B/C와 원본의 비교:

|관측|실제 수|불일치|
|---|---:|---:|
|3prefix의 generation forward|12단계|0|
|generation logits|589824 BF16좌표|0|
|전체 층 KV|760320 BF16좌표|0|
|모든 tensor layout/cache길이·필드/RNG/선택 sequence|전체 등록 실행|0|
|별도 완전 prefill logits(A)|196608 BF16좌표|0|

원본 tensorhash는 생성/실행 전후 모두
`96c04987c93051ba2608b2e0905d67e36d299873bcf606ea2787cc01a77a93db`.
원본 HF와 후보는 같은 ATen kernel을 사용하므로, 이것을 독립적인 초월함수
구현의 보편 정확성 증명으로 확대하지 않는다. 실제 공개 모델의 제한된 생성
교체 검사이며, held-out 자연어 품질/긴 문맥/다른모델은 검증하지 않았다.

## 6. 핵심 비용 — 세 후보 모두 미통과

|항목|원본|A/C 변환 후|
|---|---:|---:|
|전체 prefill native calls|3370|3340|
|전체 prefill matrixMAC|538472576|538472576|
|generation prefill matrixMAC|453537920|453537920|
|첫 decode matrixMAC|134652704|134652704|
|다음 decode matrixMAC|134687264|134687264|
|세 번째 decode matrixMAC|134721824|134721824|

동일 연산 CSE는 이 실제 그래프에서0개다. 제거된30개는 live roots에 필요 없는
보조 node다. 주요 matrixMAC은 **100% 유지**된다. 연산 node 약1% 감소를
행렬 계산 감소나 CPU/GPU 시간 향상으로 바꾸어 말하지 않는다.

A는5개의 shape/context program을 생성하며 원본forward5회를 준비에 지불한다.
C는4개를 생성해 원본forward4회를 준비에 지불한다. 현재입력이 새로운 signature면
새 capture가 필요하므로, 이 준비를 model-once 비용이라고 하지 않는다. 더 긴
문맥의 수백 graph나 CUDA백엔드를 만드는 것으로 후보를 확대하지 않았다.

총9개 JSON program은11775553B이며 원본checkpoint는 별도다. 소스는 대략1.28~1.37MB
각각이고 originaltensor273개를 참조한다. table나model압축 결과가 아니다.

B는12step모두 원본 전체forward를1회씩 호출한다. 실제 payload decode/encode의
최소RW총량은5114880B이며, 원본body연산/가중치접근/metadata/contiguouscopy/할당은
추가다. codec의 등가성은 native body를 더 싸게 갱신하는 정리가 아니다.

O4 비용 형태:

- A: checkpointload + sum_signature(captureF_W + 변환 + 저장/로드) +
  sum_steps(모든 남은nativekernel + maskguard + dispatch + cacheupdate).
- B: checkpointload + sum_steps(decodeKV + **F_W** + encodeKV + metadata).
- C: A의준비비용 + sum_steps(TOPdemand탐색 + 모든필수nativekernel + 참조관리).

logical native result/live bytes는 graph shape를 세어 얻은 값이다. backendworkspace,
allocator, cache, physicalbus traffic, CPU벽시계 또는 대상GPUpeak의 실측이 아니다.
원본streaming을 없앴다는 주장도, 10x나4BQ4비교주장도 하지 않는다.

## 7. 재현과 남은 의무

### 저장 형식 보존
원본 `requirements.observed.txt`는 BOM·CRLF를 포함한 관측 바이트를 유지한다.
실행 편의를 위한 `requirements.replay.txt`만 별도로 UTF-8/LF로 생성한다.
native 생성기의 graph_*_before/after 텍스트 끝 공백도 원본 해시와 함께
보존하며 해당 캡처 파일만 Git 텍스트 diff에서 제외한다. 직접 작성한 Python과
문서는 계속 공백 검사를 받는다. 재실행 복사본은 Git에서 제외하고 canonical
run_v3의 실제 logits/KV/RNG binary는 해시와 함께 직접 보존한다.
과거 root 문서 snapshot은 parent commit의 Git blob 원본 바이트와 일치한다.
검증기는 `--receipt`로 새 경로를 받아 기존 결과나 검증 기록을 덮어쓰지 않는다.

`verify.py --freeze`가 최초 run_v3의3510sciencefile을 고정했다. 이후
`verify.py --replay results/replay_01`로 실제 모델/graph를 다시 생성하고
14tests와3510file의 일치를 확인했다. JSON의 build/load시간3개 필드만 변동으로
제외하고, 원래 시간관측은 파일에 보존한다. timing재현성을 주장하지 않는다.

원본download/venv/재실행복사본은Git에서 제외한다. 실제inputs/output/KV/RNG,
codeprograms, 원래오류, 조건수정, source와manifest는 보존한다.

O1/O2/O3는 이 한 checkpoint/ABI의 제한형 실행에 대한 일부 구성이며 보편 의무는
OPEN이다. O4의 실제 비용 inventory는 확보했으나 빠른충분상한이 없다. O5는
미충족·목표장비미실측, O6는 이 corpus의 재검증이 가능하다. 전체mission O1–O6을
완료로 올리지 않는다. 기존output_envelope/correlation_source를 재개하지 않는다.

다음 핵심은 **원본nativekernel을 호출하기 전에** 그 kernel이 제공할 정확한
정보를 훨씬 적은 읽기·산술로 생산하는 유한primitive다. 현재 A/B/C에 더 큰
graph, codec최적화, TOP반복횟수, observer제외를 추가하는 것은 그 발견이 아니다.
후속 연구는 다시 세 materially distinct원리의 실제producer/10x경로/전비용을
먼저 비교해야 한다. 이론과목표를낮추거나 후보탈락을전체불가능성으로말하지 않는다.

## 8. 참고한 일차 자료와 실제 소스

- 원본 checkpoint: https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/93efa2f097d58c2a74874c7e644dbc9b0cee75a2
- PyTorchFX: https://docs.pytorch.org/docs/stable/fx.html
- PyTorch make_fx: https://docs.pytorch.org/docs/stable/generated/torch.fx.experimental.proxy_tensor.make_fx.html
- HF4.55.4 cache: https://huggingface.co/docs/transformers/v4.55.4/en/internal/generation_utils
- NVIDIA floatingpoint: https://docs.nvidia.com/cuda/floating-point/index.html
- 사용한 실제HF cache/masking/model소스: `results/reference_sources/`와 sourcehash기록.

문헌의 개념을 새로 발명했다고 주장하지 않는다. 실제 제한형nativeconstructor와
완전state관측을 만들었다는 것이 이번 구성의 범위다.

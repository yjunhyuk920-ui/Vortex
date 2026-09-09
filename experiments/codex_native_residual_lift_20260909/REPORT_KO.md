# 두 native 질의로 반올림 잔차 복원

선언한 FP32 누산·BF16 저장 조건에서, 첫 결과가 버린 낮은 비트를 두 번째
인과적 질의로 정확히 복원하는 유한 절차를 구성했다. 계수는 질의 사이에
바뀌지 않는다. 정수256과257의 첫 BF16 출력 충돌을 실제로 재현했고,
두 번째 질의 후 둘을 구분했다. 증명과 비용은 PROOF.md, ACCOUNTING.json.

가속 코어는 아니다. 전체 augmented MatVec를 두 번 지불하며, 등록된405B
형상에 넣을 수 있는 독립 원본 정보는 약1.0877GiB라 전역8GiB 상태에
전부 들어간다. 전역 비선형 표현을 배제하는 하한으로 사용할 수 없다.

실행: 순서 지정 FP32 제어106개, 전체 정수 경우2,097,280+16,768개,
잘못된 입력 거부9개, 실제 CPU torch linear/mv24회. 출력 word 대조와
CPU RNG 불변 확인이다. HF 생성·KV·모든 continuation 증명은 아니다.
큰 모델 실험으로 확대하지 않았다.

미완성 하위 코드의 bias 비트 수·uint8 정답 넘침·alpha 범위 결함을
원본 보존 후 실제 재현하고 수정했다. 정본 results.json SHA256 476e04d9db6ed182feab6c4e7bb00de76a12b010765eea77fd9cc18a3b5c9bcf.

새 보편 실행 원리 세 개는 아직 구성하지 못했다. O1~O5 OPEN/O6 PARTIAL,
THEORY_STATUS=NOT_ESTABLISHED, HARDWARE_STATUS=NOT_TESTED, CORE_ADMISSION=false.
현재405B 하드웨어가 없으며 전체 목표는 미완료다.

# EXP-100A 정정 결과

전체 계산을 다시 실행했다. 기존 무효 원인을 고친 결과는 고정된 제한 검색의
후보 탈락이다. 새405B 실행기나 모든 FMM의 불가능 증명이 아니다.

- 93개 catalog 키,59개 계수 분해,128개 orientation,132개 정수 제어 검사,
  50개 shape/block 검색을 실제로 실행했다. 정수 불일치0.
- 빈 direct 셀의2,304개 계획 모두 작업공간 초과였다. 최소12.24255GiB.
  저장된 계수에서 별도 정수 합으로 확인했다.
- 최선 direct 산술38.25165%, 무료 변환 oracle13.01026%가 남는다.
  둘 다 첫10% gate 미달. 현재 수정 전후의 비용과 선택 계획은 정확히 같다.
- 미완료·계수 불일치·허용 계획 유실은 계속 무효로 처리한다.
- 무료 미래 입력·native repair·전처리 가정은 남아 있다. 전체 비용 상한,
  native 출력/RNG/KV 증명,405B/CUDA/8GiB/지연 검증은 달성하지 않았다.

최종 정본 hardened_run/result.json SHA256 375a4f4a9d5055429664f0c042c02f2d128fbf2cd8196b1fcdc6ffa87980a121.
O1~O5 OPEN/O6 PARTIAL, THEORY_STATUS=NOT_ESTABLISHED,
HARDWARE_STATUS=NOT_TESTED, CORE_ADMISSION=false. 사용자에게405B 하드웨어가 없다.
새 원리 세 개의 구성과 전체 목표는 미완료이며 전역 구성 연구를 계속한다.

독립 검토의 짝지은 평가 누락을 실제 실패 사례로 재현하고 보강했다.
최종 17개 focused 검사가 통과했으며 전체 gate를 다시 실행했다.
직접 평가84,712개와 oracle15,608개가 상태 수에서 계산한 기대 수와 같다.
비용·선택 결과는 세 번의 현재 전체 실행에서 동일하다.

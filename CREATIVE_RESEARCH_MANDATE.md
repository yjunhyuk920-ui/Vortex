모든 핵심 연구 회차는 기존 기법의 조합·미세변형을 시작점으로 삼지 말고, 현재 병목을 만드는 숨은 전제·계산 순서·정보 흐름·검증 단위 중 최소 하나를 뒤집은 서로 다른 새 실행 원리 3개를 먼저 발명한 뒤, 각 원리가 원본 dense 연산이나 weight 이동을 10× 이상 제거·상각할 수 있는지를 정확성·자원식·최저비용 반증 Gate로 비교하여 가장 강한 하나만 구현하라.

연산 운영은 `샌드박스 우선, GitHub 최종`으로 고정한다. 수식·탐색·프로토타입·반례·단위시험·빠른 수리는 먼저 샌드박스에서 반복하고, 가장 싼 Gate를 통과한 생존 후보·결정적 음성 결과·재사용 가능한 연구 인프라만 원격 branch에 고정한다. GitHub Actions는 독립 재현, 고정 public checkpoint, 장시간 실행, artifact·checksum 보존에 사용하며 매 탐색 반복을 workflow로 만들지 않는다. 샌드박스 결과만으로 연구 완료를 선언하지 않고, 원격 결과 commit과 SHA read-back 이후에만 `REMOTE_COMMIT_VERIFIED`를 사용한다.

매 의미 있는 repository 회차에서 `README.md`의 목표, 작업 방식, 활성 frontier, 최신 권위 결과, quick start, repository map, 광고된 기능·test count가 현재 사실과 일치하는지 확인한다. 바뀐 내용이 있으면 같은 회차에 README를 갱신하고, 바뀐 내용이 없으면 최종 보고에 구체적인 `README_UNCHANGED_REASON`을 남긴다.

모든 핵심 연구 회차는 기존 기법의 조합·미세변형을 시작점으로 삼지 말고, 현재 병목을 만드는 숨은 전제·계산 순서·정보 흐름·검증 단위 중 최소 하나를 뒤집은 서로 다른 새 실행 원리 3개를 먼저 발명한 뒤, 각 원리가 원본 dense 연산이나 weight 이동을 10× 이상 제거·상각할 수 있는지를 정확성·현실적 자원식·최저비용 반증 Gate로 비교하여 가장 강한 하나만 구현하라.

연구 운영은 `로컬 연구·로컬 검증·GitHub 커밋 인계`로 고정한다. 수식·탐색·프로토타입·반례·단위시험·public checkpoint 실행·raw evidence·checksum 생성까지 로컬/샌드박스에서 완료한다. 로컬 검증이 끝난 회차는 GitHub에서 같은 실험을 다시 돌리지 않는다. GitHub는 source/config/result/log/checksum/docs/README를 commit·push하고 원격 SHA를 읽어 다른 세션으로 인계하는 용도로만 사용한다.

기본 상태 흐름:

```text
LOCAL_RESEARCH
-> LOCAL_VALIDATION_PASS
-> COMMIT_PUSHED
-> REMOTE_COMMIT_VERIFIED
```

GitHub Actions, clean-room reproduction, hosted checkpoint rerun은 기본 요구사항이 아니다. 사용자가 명시적으로 요청한 경우에만 예외적으로 실행한다. 기존 자동 CI가 실행되더라도 로컬에서 이미 검증된 과학적 판정을 다시 승인하는 필수 Gate로 취급하지 않는다.

매 의미 있는 repository 회차에서 `README.md`의 목표, 작업 방식, 활성 frontier, 최신 권위 결과, quick start, repository map, 광고된 기능·test count가 현재 사실과 일치하는지 확인한다. 바뀐 내용이 있으면 같은 commit에 README를 갱신하고, 바뀐 내용이 없으면 최종 보고에 구체적인 `README_UNCHANGED_REASON`을 남긴다.

## Reality-first authoritative execution

Every new core Gate has one authoritative arm: `REAL_EXECUTOR_ONLY`.
Future target tokens or hidden states, perfect selectors, free `N/A`, free transforms, free metadata/workspace, free repair/fallback, unmeasured compression, and peak throughput presented as sustained throughput are forbidden from satisfying a promotion threshold. Synthetic or target-seeing calculations may appear only as non-authoritative debugging diagnostics.

The authoritative arm must execute a finite-word causal path and charge candidate generation, every target position, verification, mismatch repair, rollback, fallback, transforms, packing, metadata, storage/host/device bytes, KV/cache, workspace, fragmentation, and measured wall time where available. Missing target quantities remain `NOT TESTED`; they are never replaced by an ideal grant.

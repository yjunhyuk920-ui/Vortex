# Causal/global producer frontier 결과 요약

이번 회차의 결론은 **미션 완성 아님**이다. 대신 causal reachability의 핵심 빈칸과
제한 family의 실제 producer를 구성했고, 그 결과 P2/P3의 남은 문제를 더 정확하게
고정했다.

## 새로 성립한 것

1. 실제 `transformers.LlamaForCausalLM` + `DynamicCache`에서 임의 binary
   `v_proj` 정보가 합법적인 token prefix를 통해 successor KV에 bit-exact하게 노출된다.
2. GQA `32 x 224`에서도 7168개 source/cache 좌표가 `0 mismatch`였다.
3. 이 제한된 basis-column family에는 dense forward를 전혀 호출하지 않는 유한한
   lossless producer를 실제로 만들었다. 2-layer `32 x 224`에서 runtime source payload는
   128 bit/token, variable-v_proj full BF16 read는 229376 bit/token으로,
   해당 제한 source에 한해 99.944%를 제거했다.
4. 표준 causal mask를 그대로 사용하면서 서로 다른 right factor를 한 trace에 넣는
   sign/delta compiler를 만들었다. 8-bit 32-query trace에서 8192개 모든 binary output
   row를 검사했고 parity decode mismatch가 0이었다.
5. 기존 nonlinear-router frontier의 정확한 area-5400 모양인 `25 x 216`에서도
   4-query native trace까지 sign/공통 magnitude, full/incremental logits, left-mask parity가
   모두 0 mismatch였다.

## 아직 성립하지 않은 것

- `25 x 216`에서 32-query 전체 native trace는 실행하지 않았다.
- 따라서 이전 strong independent-32 union theorem을 batch-1 Transformer의 완전한
  native theorem으로 승격하지 않는다.
- 8 GiB global advice가 matrix/layer 사이를 임의 nonlinear하게 섞는 경우의 direct-sum
  lower bound도 없다.
- binary/basis family producer는 임의 Q4/BF16 checkpoint producer가 아니다.

## P2 global producer

명시적인 arbitrary-checkpoint `encoder -> adaptive address -> exact decoder`는 아직 없다.
Boolean MatVec의 알려진 빠른 succinct data structure는 randomized/errorful Boolean
semiring이고, deterministic 경로 역시 native Q4/BF16 ordered arithmetic을 계산하지 않는다.
binary bitpack/popcount를 Q4/BF16로 올리면 결국 모든 원본 bit plane을 읽어야 하므로
목표 회피가 아니다.

## P3 dynamic summary

새 causal compiler는 연속 query의 right factor를 서로 완전히 다르게 만들 수 있다.
`y=W s`를 들고 있다가 column delta로 갱신하는 방식은 Hamming distance가 `n`인 legal
query 쌍에서 모든 column을 다시 만지므로 기존 response-column transport로 되돌아간다.
따라서 P3가 성공하려면 arbitrary dense column 집합을 subdense하게 합치는 새 producer가
필요하며, 그 핵심은 다시 P2의 exact MatVec data structure 문제다.

## 상태

```text
THEORY_STATUS   NOT_ESTABLISHED
HARDWARE_STATUS NOT_TESTED
O1 OPEN
O2 OPEN
O3 OPEN
O4 OPEN
O5 OPEN
O6 PARTIAL
```

405B, CUDA, <=8 GiB target GPU, PCIe/SSD/HBM, native 4B Q4 p50/p95, TTFT는 이번
회차에서도 실행하지 않았으며 `NOT TESTED`를 유지한다.

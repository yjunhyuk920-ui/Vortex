# Principle III independent gauge review

검토 기준은 읽기 전용 원본 `C:\ChatOnSteroids\Vortex-output-envelope`의 concurrent HEAD `dab57747b690ecc4ba0562f3af39e781af01a836`이다. 현재 HEAD와 지정 SHA가 같음을 확인했다. 원본 수정, 실험 재실행, GPU/원격 연산은 하지 않았다. 요청 모델/추론 강도는 `gpt-5.6-sol/high`이지만 실제 선택을 확인할 도구 메타데이터가 없으므로 **미확인**이다.

## 판정

1. **일반화 정리: 정확함.** 체 $F$ 위의 가역 $A,B,C\in F^{n\times n}$가 모든 $x,y$에 대해
   \[
   C(x\mathbin{.*}y)=(Ax)\mathbin{.*}(By)
   \]
   를 만족할 필요충분조건은 하나의 같은 permutation matrix $P$와 비영 대각행렬 $D_a,D_b$가 존재하여
   \[
   A=D_aP,\qquad B=D_bP,\qquad C=D_aD_bP
   \]
   인 것이다. 따라서 세 wire gauge는 일반 체에서 서로 다를 수 있지만, 서로 다른 좌표 순열은 허용되지 않는다. 차이는 두 독립 대각 스케일뿐이며 출력 스케일은 그 좌표별 곱으로 고정된다.

2. **GF(2) 특수화: 정확함.** 유일한 비영 스칼라가 1이므로 $D_a=D_b=I$, 따라서 $A=B=C=P$이다.

3. **개별 output rank의 합을 전역 $n^2$ 곱 하한으로 쓰는 주장: 실패.** 각 output의 bilinear matrix rank는 맞지만, output들이 같은 $n$개의 중간곱을 공유한다. 합산은 그 공유를 중복 계산한다.

4. **native permutation arm의 현재 문구: schedule 조건이 빠져 있어 과도함.** 좌표 순열만으로 elementwise 연산은 보존되지만, dense dot product의 열 순열은 finite-word reduction 묶음과 순서를 바꿀 수 있다. 원래 leaf slot과 reduction tree/order까지 운반하거나 그대로 에뮬레이션해야 arbitrary finite-word exactness를 주장할 수 있다.

5. **비단항 transformed gate 불가능 주장: 확대하면 틀림.** raw-word XOR bijection처럼 field 정리의 가정을 벗어나는 빠른 가역 circuit은 decode–native multiply–encode 방식의 유한한 transformed gate를 준다. 분류 정리는 raw coordinatewise product를 변환 없이 그대로 유지하는 field-linear gauge만 제한한다.

6. **미션 범위 영향(최종 판정 아님):** 이 결과는 Principle III의 field/GF(2) 필요조건과 한 잘못된 rank 하한을 정리하는 보조 정리다. 실제 SwiGLU, native finite-word 전체 상태, 405B/8 GiB/지연 목표를 닫는 근거를 제공하지 않는다. 전체 코드/증명 및 미션 판정은 주 에이전트 범위다.

## 증명

출력 행을 $i$라 하자. 양변에서 $x_jy_k$의 계수를 비교하면
\[
A_{ij}B_{ik}=0\quad(j\ne k),\qquad C_{ij}=A_{ij}B_{ij}.
\]
가역성 때문에 $A,B$의 각 행은 비어 있지 않다. 행 $i$에서 $A_{ij}\ne0$, $B_{ik}\ne0$인 지수들을 하나씩 고르면 첫 식과 체의 무영인자 성질 때문에 반드시 $j=k$이다. 한쪽 행의 다른 비영 위치도 반대쪽의 고정된 비영 위치와 비교하면 같은 위치여야 한다. 따라서 두 행의 support는 공통 singleton $\{\sigma(i)\}$이다. 가역성은 $\sigma$가 permutation임을 강제한다. 그 위치의 값을 $a_i,b_i\ne0$라 두면 $C_{i,\sigma(i)}=a_ib_i$이므로 표시한 세 행렬 형태가 나온다. 역방향은 직접 대입하면 성립한다.

특히 automorphism 식 $E(x.*y)=(Ex).*(Ey)$에서는 $A=B=C=E$이므로 각 대각값 $d_i$가 $d_i=d_i^2$를 만족한다. $d_i\ne0$이어서 모든 체에서 $d_i=1$, 즉 $E=P$이다. 이 결론은 체 가정에 의존하며 영인자가 있는 일반 환으로 자동 확장되지 않는다.

## rank 합산 반례와 실제 비용

$c_i^T$를 $C$의 $i$번째 행이라 하면 encoded 입력 $z=Ax,w=By$에서 output $i$는
\[
z^T T_iw,\qquad T_i=A^{-T}\operatorname{diag}(c_i)B^{-1}.
\]
좌우 가역곱은 rank를 보존하므로
\[
\operatorname{rank}(T_i)=\operatorname{rank}(\operatorname{diag}(c_i))=\operatorname{nnz}(c_i)
\]
는 맞다. 그러나 이를 output별로 더한 값은 vector-valued bilinear map의 tensor rank 하한이 아니다.

구체적으로 $F=\mathbb R$, $A=B=I_2$,
\[
C=\begin{bmatrix}1&1\\1&2\end{bmatrix}
\]
이면 두 $T_i$가 모두 rank 2라 합은 4이다. 하지만 $p_1=z_1w_1,p_2=z_2w_2$ 두 variable-by-variable 곱만 계산한 뒤 $(p_1+p_2,\ p_1+2p_2)$를 반환하면 된다. 일반적으로 정확한 공유 decoder는
\[
C\big((A^{-1}z).*(B^{-1}w)\big)
\]
이며 bilinear 중간곱은 $n$개다. 실제 global tensor rank도 가역 세 factor의 flattening으로 하한 $n$, 위 decoder로 상한 $n$이어서 정확히 $n$이다.

이 반례가 비용이 작다는 뜻은 아니다. $A^{-1}z$, $B^{-1}w$, $Cp$가 dense이면 최대 세 번의 dense matvec, 해당 계수 저장/읽기, multiply-add/reduction, 이동, 임시 버퍼, 주소, workspace가 든다. 인접 weight에 gauge를 흡수하면 그 비용과 native schedule 의무가 인접 producer/consumer로 이동할 뿐 사라지지 않는다. 따라서 $n^2$ **bilinear 곱 하한**은 기각하되, dense linear transform의 완전 유료 상한으로 후보를 별도 평가해야 한다.

## native reduction schedule 반례

BF16에 정확히 표현되는 한 행
\[
W=[2^{24},1,-2^{24},0],\qquad x=[1,1,1,1]
\]
을 생각한다. separate products 뒤 FP32 round-to-nearest-even balanced tree가 원래 logical slot 순서로
\[
\operatorname{fl}(\operatorname{fl}(2^{24}+1)+\operatorname{fl}(-2^{24}+0))
\]
를 계산하면 `fl(2^{24}+1)=2^{24}`이어서 결과는 0이다. 열과 입력을 함께 `[0,2,1,3]` 순서로 바꾼 뒤 새 물리 위치의 표준 balanced tree를 실행하면
\[
\operatorname{fl}(\operatorname{fl}(2^{24}-2^{24})+\operatorname{fl}(1+0))=1.
\]
수학적 dot product와 leaf 값은 같아도 native 결과는 다르다.

따라서 $W'=P_{out}WP_{in}^{-1}$만으로는 충분하지 않다. 각 원래 leaf가 이동한 새 물리 열을 기록하고, 그 값을 **원래 logical leaf slot**에 놓아 원래 reduction tree, 연산 순서, 반올림/변환 지점을 그대로 실행해야 0이 복구된다. backend가 이를 보장하지 못하면 exact reference schedule 에뮬레이션 또는 유료 fallback이 필요하다. 이 mapping, gather/scatter, schedule metadata와 kernel 비용도 전부 상한에 포함해야 한다.

## 실제 SwiGLU와 native 주장 범위

분류 정리는 field의 bilinear Hadamard node에만 직접 적용된다. 실제 SwiGLU의 한 branch에는 SiLU가 있고 native 산술에는 반올림, overflow, 특수값과 backend reduction/fusion schedule이 있다. 일반 $A$에 대해 $A\,\mathrm{SiLU}(A^{-1}z)$는 원래 coordinatewise SiLU가 아니며, 대각 스케일도 일반적으로 $\mathrm{SiLU}(at)=a\,\mathrm{SiLU}(t)$를 만족하지 않는다. 비단항 gauge는 transformed activation/Hadamard를 명시적으로 실행하고 비용을 내야 한다.

같은 permutation을 두 product branch와 output에 맞추면 coordinatewise SiLU/Hadamard 자체는 단순 재배치로 보존된다. 그러나 그 전후 dense producer/consumer의 열 방향 순열은 위 schedule 조건을 만족해야 한다. 그러므로 현재 근거로 주장 가능한 범위는 다음뿐이다.

- field/GF(2) 분류 정리와 non-monomial gauge가 raw coordinatewise Hadamard를 그대로 유지할 수 없다는 필요조건;
- schedule을 포함한 명시적 finite-word program이 있을 때의 permutation-based exactness 후보;
- arbitrary native checkpoint에 대한 실제 SwiGLU exactness, logits/output, RNG 소비, next state, KV의 모든 continuation 보존은 미증명;
- CPU/RAM/SSD/PCIe/HBM, compile/storage, address/decoder/verification/fallback, peak 8 GiB와 latency의 완전 유료 상한은 미제시·미검증.

## raw-word XOR gauge가 정리의 불가능 범위를 제한함

검토 입력으로 제공된 이 보조구성은 지정 SHA에는 아직 artifact가 없으므로, 아래 판정은 구성 설명에 대한 정적 수학 검토다. 길이 $n=2^d$인 16/32-bit raw-word 벡터에 subset-zeta butterfly $E$를 적용하는 구성은 타당하다. 각 stage는 서로 겹치지 않는 $n/2$개 word XOR shear이고, 같은 stage를 두 번 적용하면 원상복구된다. 따라서 전체 inverse는 stage를 역순으로 적용하는 유한 algorithm이다. word를 부동소수 값으로 해석하지 않고 raw bits로만 XOR하면 모든 bit pattern을 정확히 복구한다.

이때 transformed Hadamard를
\[
G_E(z,w)=E\!\left(\operatorname{native\_mul}(E^{-1}z,E^{-1}w)\right)
\]
로 정의하면 원래 raw words를 먼저 복구한 뒤 원래 native multiply를 실행하므로 특수값을 포함한 그 primitive의 결과를 정확히 재인코딩한다. 산술 연산 수는 두 inverse와 한 forward에 대한
\[
3\,(n\log_2 n/2)\text{ word XOR}+n\text{ native multiply}
\]
이다. 이는 조밀한 nonmonomial bit-mixing 좌표에서도 transformed gate가 “정의 불가”이거나 반드시 $\Omega(n^2)$라는 확대 주장의 반례다.

이 구성은 위 field 정리와 모순되지 않는다. $E$는 raw-bit 공간의 GF(2)-linear bijection이지만 native floating multiply는 그 field의 coordinatewise product가 아니며, $G_E(z,w)$도 단순한 $z.*w$가 아니라 두 inverse와 forward를 명시한 conjugated program이다. encoded words는 중간에 float로 해석하면 NaN 등 임의 bit pattern일 수 있으므로 opaque bits로 취급해야 한다. SiLU, residual, norm, KV/state 등 모든 다른 소비자에도 각자의 exact conjugated program 또는 decode가 필요하다.

또한 이 count는 instruction count일 뿐 완전한 비용 상한이나 가속 증명이 아니다. 세 transform의 word reads/writes, 주소, loop/synchronization, workspace와 native multiply가 모두 유료이고, 원래 Hadamard의 $n$ multiply보다 XOR 일이 추가된다. 가장 중요한 점은 arbitrary dense projection $F_W$에 대해 $E F_W E^{-1}$를 subdense하게 실행하는 constructor가 없다는 것이다. 단순히 $E^{-1}$로 decode하여 원래 dense $F_W$를 실행하고 다시 $E$를 적용하면 정확하지만 원래 dense projection 비용을 그대로 낸다. 따라서 이 보조구성은 cheap-gate 불가능론의 범위만 좁히며 임의 checkpoint core나 속도 결과가 아니다.

## 근거와 다음 판단사항

- `dab57747b690ecc4ba0562f3af39e781af01a836:experiments/implicit_nonlinear_direct_query_20260909/PREREGISTRATION.md:191-213`: transformed nonlinear/residual을 유료화하고 permutation arm 및 general gauge extension을 정의한다.
- 같은 파일 `:217-232`: 상태 대응식과 gather/scatter, workspace, nonlinear conjugation, fallback 비용 의무를 둔다. 위 schedule transport를 exactness 식의 필수 조건으로 추가해야 한다.
- 같은 파일 `:236-249`: GF(2) automorphism lemma와 dense SwiGLU/Hadamard adversary를 명시한다. 독립 $A,B,C$ 일반화는 이 lemma를 강화하되 미션 core로 승격시키지 않는다.
- `dab57747b690ecc4ba0562f3af39e781af01a836:docs/CONSTRUCTIVE_THEORY_CONTRACT.md:15-17,23-32`: 고정 미션과 O1-O6 범위.
- 같은 파일 `:36-52`: decoder/schedule/cost 상한과 모든 비용의 유료 원칙. rank 하한은 이 상한을 대신하지 못한다.
- 같은 파일 `:66-68,84-86`: field/GF(2) 보조 결과를 native/405B/hardware 성취로 올릴 수 없다.

주 에이전트가 판단할 다음 사항은 (a) 분류 정리를 Principle III의 정확한 필요충분조건으로 채택하되 distinct gauges를 `same P + independent nonzero diagonals`로 표현할지, (b) 잘못된 output-rank 합산 하한을 폐기하고 공유 decoder의 전체 비용 상한으로 교체할지, (c) safe permutation arm의 정의에 original logical leaf/reduction schedule transport를 넣을지, (d) nonmonomial 결론을 “raw coordinatewise field product를 그대로 유지할 수 없음”으로 제한하고 explicit fast conjugation circuit의 가능성은 열어둘지이다. 이 검토가 제공한 근거만으로는 기존 `THEORY_STATUS=NOT_ESTABLISHED`, `HARDWARE_STATUS=NOT_TESTED`를 변경할 수 없다.

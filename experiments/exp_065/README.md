# EXP-065 — Real-Q4 Exact Kronecker-Rearrangement Rank Gate

This Gate examines every nontrivial ordered shape factorization of each pinned real-Q4 matrix. For `W` shaped `(m1*m2, n1*n2)`, it rearranges block vectors into a matrix shaped `(m1*n1, m2*n2)`. The rank of this rearrangement equals the minimum number of Kronecker-product terms over the field.

Each candidate receives independently verified modular-rank certificates under two primes. The certified rank is a rigorous lower bound on exact integer/rational Kronecker rank. Favorable lower-bound accounting charges 4-bit factors, both reshape multiplications, cross-term output additions, intermediates, per-row scales, biases, metadata and query bytes.

A low modular rank is not an exact factor reconstruction. Surviving lower-bound candidates may only advance to a separate reconstruction Gate. Q4 model-output preservation, physical kernels, actual Transformer operation replacement, 405B execution, 8 GiB VRAM and target hardware are not tested.

```bash
bash experiments/exp_065/reproduce.sh
```

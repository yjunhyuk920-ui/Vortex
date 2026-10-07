# Vortex source construction follow on review

Prepared 2026-10-07 UTC. These are literature dispositions, not independently rerun author experiments or new Vortex performance results.

## NEXUS

The paper builds neuromorphic logic circuits from floating-point bit representations. Its section 4.2.2 reports Qwen3-0.6B full-forward outputs with 27.7 percent zero-ULP agreement, mean 6.19 ULP difference and maximum 512 ULP difference. Appendix D.4 specifies simulation on eight A100 80 GB GPUs and model-estimated Loihi 2 energy. These results do not establish Vortex's exact output and full-state contract or single-8-GiB cost objective. The manuscript's use of the phrase bit exact does not eliminate its reported numerical differences. The disposition concerns the presented implementation and evidence, not every bit-circuit method. [Primary manuscript](https://arxiv.org/html/2601.21279v1).

## M Fibration

The weighted-fibration construction is broader than identical-expression CSE but uses a commutative-monoid network semantics. The neural-network application uses real addition; it does not prove preservation of prescribed finite-word accumulation order. Table 1 keeps 100 percent of units and parameters at epsilon zero for all three presented networks. Its compression cases use positive epsilon. These data do not reopen the excluded native quotient family or admit an exact 405B producer. [Primary manuscript, sections 8 and 9](https://arxiv.org/html/2608.25598v1).

## ReLU stability

The NeurIPS 2021 construction certifies always-active or inactive ReLU units on a specified input domain, then removes redundant units. Its real-function and ReLU-domain assumptions are insufficient for arbitrary native SwiGLU transformers and all legal continuations. The paper also states scale limitations for very large networks. This is a conditional source idea, with no target-compatible constructor supplied here. [Primary paper](https://papers.nips.cc/paper_files/paper/2021/file/e35d7a5768c4b85b4780384d55dc3620-Paper.pdf).

## Prox

Prox predicts an intermediate-channel mask using input sparsity and quantized proxy weights, then evaluates only retained channels at original precision. Its exact retained-channel evaluation does not restore the omitted nonzero channels. The paper explicitly treats the result as an accuracy-efficiency tradeoff and reports perplexity changes. No all-legal native result or state certificate is supplied. Therefore its two-stage selection is not an exact Vortex producer. No proxy or target model was run. [Primary manuscript, sections 3 and 4](https://arxiv.org/html/2607.27591v1).

## Concrete unfinished source lead

The official TMLR September 2026 index lists Mechanistic Interpretability of Transformer MLPs Through Exact Soft Gate Decomposition by Arnab Barua, Mobyen Uddin Ahmed and Shahina Begum. The public abstract describes input-dependent effective matrices, including gated SiLU TinyLlama. This is a lead for inspection, not an admitted acceleration candidate.

The next decisive questions are whether the gate and effective matrix are constructed before the original dense projections, whether their construction is cheaper after all costs, and whether its equivalence is native bit and state exact. If producing the effective matrix first requires the original dense forward or a higher-cost input-dependent matrix product, it has not supplied the missing causal source. A literal dense effective-matrix implementation must pay its own formation, not just the later matrix-vector multiplication.

The verified publisher PDF and forum links currently return browser verification rather than manuscript text. No verification challenge, sign-in or permission agreement was completed. The full manuscript's algorithm and numerical protocol remain UNREAD and UNVERIFIED. No conclusion about its exact constructor is invented from the abstract.

Sources: [TMLR index](https://jmlr.org/tmlr/papers/), [verified paper link](https://openreview.net/pdf?id=20lTANRDAz), [verified forum](https://openreview.net/forum?id=20lTANRDAz).

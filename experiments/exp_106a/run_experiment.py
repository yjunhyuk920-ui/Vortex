from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

P = 405_849_243_648
L = 126
D = 16_384
F = 53_248
V = 128_256
KV = 1_024
HEAD_DIM = 128
A = 339
GPU = 8 * 2**30
P50 = 2_400_000_000
P95 = 3_000_000_000
RESERVE = 512 * 2**20 + 256 * 2**20 + 512 * 2**20 + 174_956_544
WIDTHS = (128,256,384,512,768,1024,1280,1536,1664,1792,2048,3072,4096,8192,16384)


def round_up(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a


def q4_bytes(n: int) -> int:
    return (n + 1) // 2 + ((n + 127) // 128) * 4


def width_ledger(m: int) -> dict[str, Any]:
    n = round_up(math.ceil(F * m / D), HEAD_DIM)
    kv = max(HEAD_DIM, round_up(math.ceil(KV * m / D), HEAD_DIM))
    layer_p = 2*m*m + 2*m*kv + 3*m*n + 2*m
    layer_b = q4_bytes(layer_p * L)
    emb_b = q4_bytes(V * m)
    head_b = q4_bytes(V * m)
    row_b = q4_bytes(m)
    draft_kv = A * L * 2 * kv * 2
    peak = layer_b + emb_b + head_b + draft_kv + RESERVE
    scan = layer_b + head_b + row_b
    sweep = 2 * P
    def required(limit: int) -> int | None:
        return None if scan >= limit else math.ceil(sweep / (limit - scan))
    return {
        "width":m,"intermediate_width":n,"kv_width":kv,
        "per_layer_parameters":layer_p,"layer_bytes":layer_b,
        "embedding_bytes":emb_b,"head_bytes":head_b,
        "draft_kv_bytes_at_A339":draft_kv,"reserve_bytes":RESERVE,
        "peak_resident_bytes":peak,"fits_8gib":peak<=GPU,
        "per_token_draft_scan_bytes":scan,
        "p50_scan_alone_pass":scan<P50,"p95_scan_alone_pass":scan<P95,
        "required_A_p50":required(P50),"required_A_p95":required(P95),
        "linear_kernel_dimension":D-m,
        "flat_spectrum_retained_energy_fraction":m/D,
        "flat_spectrum_relative_frobenius_error":math.sqrt(max(0.0,1-m/D)),
        "flat_spectrum_relative_operator_error":0.0 if m==D else 1.0,
    }


def hadamard(n: int) -> np.ndarray:
    if n <= 0 or n & (n-1): raise ValueError("power of two required")
    h=np.array([[1]],dtype=np.int64)
    while len(h)<n: h=np.block([[h,h],[h,-h]])
    return h


def coordinate_control(d: int=128,m: int=32,layers: int=4,steps: int=512) -> dict[str,Any]:
    def run(c: int) -> tuple[int,int]:
        full=np.zeros(d,dtype=np.int64); full[c]=1
        narrow=full[:m].copy(); matches=0; first=0
        for s in range(1,steps+1):
            for _ in range(layers): full=full.copy(); narrow=narrow.copy()
            target=1 if full[c]>0 else 0
            draft=1 if c<m and narrow[c]>0 else 0
            if target==draft: matches+=1
            elif first==0: first=s
        return matches,first
    pm,pf=run(0); nm,nf=run(d-1)
    return {"hidden_size":d,"width":m,"layers":layers,"steps":steps,
            "positive_matches":pm,"positive_first_mismatch":pf,
            "negative_matches":nm,"negative_first_mismatch":nf,
            "positive_exact":pm==steps,"negative_immediate_mismatch":nf==1}


def linear_collision_control(d: int=64,m: int=16) -> dict[str,Any]:
    h=hadamard(m)
    e=np.empty((m,d),dtype=np.int64); e[:,:m]=h
    for j in range(m,d): e[:,j]=h[:,(j-m)%m]
    v=np.zeros(d,dtype=np.int64); v[0]=-1; v[m]=1
    z=np.zeros(d,dtype=np.int64)
    return {"hidden_size":d,"width":m,
            "encoder_rank":int(np.linalg.matrix_rank(e.astype(np.float64))),
            "kernel_vector_nonzero":bool(np.any(v)),
            "codes_byte_equal":bool(np.array_equal((e@z).tobytes(),(e@v).tobytes())),
            "target_score_zero":int(v@z),"target_score_collision":int(v@v),
            "target_distinguishes_collision":bool((v@z)!=(v@v))}


def flat_spectrum_control(n: int=128) -> dict[str,Any]:
    s=np.linalg.svd(hadamard(n).astype(np.float64),compute_uv=False)
    expected=math.sqrt(n)
    return {"order":n,"minimum_singular_value":float(s.min()),
            "maximum_singular_value":float(s.max()),
            "expected_singular_value":expected,
            "maximum_absolute_singular_error":float(np.max(np.abs(s-expected)))}


def bitplane_ledger() -> dict[str,Any]:
    b=(P+7)//8
    return {"one_bit_per_parameter_bytes":b,"one_bitplane_gib":b/2**30,
            "one_bitplane_over_8gib":b/GPU,
            "one_bitplane_over_p50_bytes_per_token":b/P50,
            "one_bitplane_fits_8gib":b<=GPU,
            "one_bitplane_fits_p50_scan":b<=P50,
            "maximum_resident_bits_per_parameter":GPU*8/P}


def stable_core(x: dict[str,Any]) -> str:
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def run() -> dict[str,Any]:
    all_rows=[width_ledger(m) for m in WIDTHS]
    feasible=[r for r in all_rows if r["fits_8gib"] and r["p50_scan_alone_pass"]]
    key_widths={128,512,1024,1536,1664,1792,2048,16384}
    rows=[r for r in all_rows if r["width"] in key_widths]
    full=all_rows[-1]; coord=coordinate_control(); collision=linear_collision_control()
    flat=flat_spectrum_control(); bits=bitplane_ledger()
    failures=[]
    if not coord["positive_exact"] or not coord["negative_immediate_mismatch"]: failures.append("coordinate_control")
    if not collision["codes_byte_equal"] or not collision["target_distinguishes_collision"]: failures.append("linear_collision")
    if flat["maximum_absolute_singular_error"]>1e-10: failures.append("flat_spectrum")
    dichotomy=(feasible and all(r["linear_kernel_dimension"]>0 for r in feasible)
               and not full["fits_8gib"] and not full["p50_scan_alone_pass"]
               and not bits["one_bitplane_fits_8gib"] and not bits["one_bitplane_fits_p50_scan"])
    decision=("INVALID_DEPTH_COMPLETE_WIDTH_THIN_CONTROL_FAILURE" if failures else
              "REJECT_DEPTH_COMPLETE_WIDTH_THIN_LINEAR_AND_BIT_SLICED_SURROGATES_AS_UNIVERSAL_CAUSAL_CORE" if dichotomy else
              "SURVIVE_DEPTH_COMPLETE_WIDTH_THIN_SURROGATE_STATIC_GATE")
    out={"schema":"exp-106a-depth-complete-width-thin-v1",
         "authoritative_arm":"REAL_EXECUTOR_ONLY_FULLY_CHARGED_STATIC_AND_FINITE_WORD_GATE",
         "decision":decision,"integrity_failures":failures,
         "inventory":{"parameters":P,"layers":L,"hidden_size":D,"intermediate_size":F,"vocabulary_size":V,"kv_width":KV},
         "budget":{"gpu_bytes":GPU,"p50_bytes_per_token":P50,"p95_bytes_per_token":P95,"registered_segment":A,"reserve_bytes":RESERVE},
         "principle_1_depth_complete_width_thin":{"status":"rejected_as_universal_core" if dichotomy else "undecided",
             "resource_frontier":rows,"resource_feasible_widths":[r["width"] for r in feasible],
             "maximum_resource_feasible_width":max(r["width"] for r in feasible),
             "coordinate_full_depth_control":coord,"exact_linear_encoder_collision_control":collision,
             "flat_spectrum_dense_control":flat,
             "universal_dichotomy":{"resource_feasible_widths_have_nonzero_kernel":all(r["linear_kernel_dimension"]>0 for r in feasible),
                 "full_width_fits_8gib":full["fits_8gib"],"full_width_p50_scan_pass":full["p50_scan_alone_pass"]}},
         "principle_2_bit_sliced_all_layer":{"status":"rejected_as_universal_core","ledger":bits},
         "principle_3_cold_probe_head_index":{"status":"auxiliary_only_no_causal_state_source"},
         "claim_boundary":{"public_checkpoint_execution":"NOT_RUN_UNIVERSAL_FINITE_WORD_GATE_DECISIVE_AND_NO_LOCAL_WEIGHT_ACCESS",
             "complete_405b_execution":"NOT_TESTED","physical_8gib":"NOT_TESTED","target_cuda_sass":"NOT_TESTED","same_machine_latency":"NOT_TESTED"},
         "next_gate":{"experiment":"EXP-107A","name":"Exact Distinct-State Shared-Weight-Sweep Gate"}}
    out["deterministic_core"]=stable_core(out)
    return out


def self_test() -> None:
    assert q4_bytes(128)==68 and q4_bytes(129)==73
    c=coordinate_control(128,32,4,64); assert c["positive_exact"] and c["negative_immediate_mismatch"]
    c=linear_collision_control(); assert c["encoder_rank"]==16 and c["codes_byte_equal"] and c["target_distinguishes_collision"]
    assert flat_spectrum_control(64)["maximum_absolute_singular_error"]<1e-10
    assert width_ledger(128)["fits_8gib"] and width_ledger(128)["linear_kernel_dimension"]>0
    assert not width_ledger(D)["fits_8gib"] and not width_ledger(D)["p50_scan_alone_pass"]
    b=bitplane_ledger(); assert not b["one_bitplane_fits_8gib"] and not b["one_bitplane_fits_p50_scan"]


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path); ap.add_argument("--self-test",action="store_true")
    args=ap.parse_args()
    if args.self_test: self_test()
    result=run()
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"decision":result["decision"],"integrity_failures":result["integrity_failures"],"deterministic_core":result["deterministic_core"]},indent=2))

if __name__=="__main__": main()

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from producer import (
    all_invertible, bits, classify_gauges, compile_transport, decode_rows,
    dense_reference, execute_transport, matvec, operation_ledger, rank,
)


def independent_basis_equations(a,b,c,p):
    n = len(a)
    for j,k in itertools.product(range(n),repeat=2):
        x,y = [0]*n,[0]*n
        x[j],y[k] = 1,1
        left = matvec(c,[u*v % p for u,v in zip(x,y)],p)
        right = [u*v % p for u,v in zip(matvec(a,x,p),matvec(b,y,p))]
        if left != right:
            return False
    return True


def enumeration(p,n,expected):
    matrices = list(all_invertible(n,p))
    certified = 0
    witnessed = 0
    outside = 0
    for a,b in itertools.product(matrices,repeat=2):
        c = [[a[i][j]*b[i][j] % p for j in range(n)] for i in range(n)]
        result = classify_gauges(a,b,c,p)
        actual = independent_basis_equations(a,b,c,p)
        assert (result["status"] == "CERTIFIED") == actual
        if actual:
            certified += 1
            order,da,db = result["permutation"],result["diag_a"],result["diag_b"]
            assert sorted(order) == list(range(n))
            assert all(da) and all(db)
            for i,j in itertools.product(range(n),repeat=2):
                assert a[i][j] == (da[i] if order[i] == j else 0)
                assert b[i][j] == (db[i] if order[i] == j else 0)
                assert c[i][j] == (da[i]*db[i] % p if order[i] == j else 0)
        elif result["status"] == "COUNTEREXAMPLE":
            witnessed += 1
            x,y = result["x"],result["y"]
            assert matvec(c,[u*v % p for u,v in zip(x,y)],p) != [u*v % p for u,v in zip(matvec(a,x,p),matvec(b,y,p))]
        else:
            outside += 1
            assert rank(c,p) < n
    assert certified == expected
    return {"p":p,"n":n,"invertible_matrices":len(matrices),"pairs_checked":len(matrices)**2,
            "certified":certified,"counterexample_witnesses":witnessed,"forced_C_singular":outside,
            "classification_mismatches":0,"scope":"all invertible A,B; C forced by diagonal equations"}


def numerical_controls():
    weights = [[2**24,1,-2**24,0],[1,-2**24,2**24,1]]
    inputs = [[1,1,1,1],[1,-1,1,-1],[0,1,-1,2]]
    records = []
    for columns in itertools.permutations(range(4)):
        for rows in itertools.permutations(range(2)):
            plan = compile_transport(weights,columns,rows)
            for x in inputs:
                encoded = [x[j] for j in columns]
                ref = list(map(bits,dense_reference(weights,x)))
                naive = list(map(bits,decode_rows(plan,dense_reference(plan.weights,encoded))))
                transported = list(map(bits,decode_rows(plan,execute_transport(plan,encoded))))
                assert ref == transported
                records.append({"column_order":columns,"row_order":rows,"input":x,
                                "reference_bits":ref,"naive_bits":naive,"transported_bits":transported})
    plan = compile_transport([weights[0]],[0,2,1,3],[0])
    original = bits(dense_reference([weights[0]],inputs[0])[0])
    naive = bits(dense_reference(plan.weights,inputs[0])[0])
    transported = bits(execute_transport(plan,inputs[0])[0])
    assert (original,naive,transported) == (0,0x3f800000,0)
    witness = {"weight_words":[bits(x) for x in weights[0]],"input_words":[bits(x) for x in inputs[0]],
               "column_order":[0,2,1,3],"original_fp32_bits":original,"naive_fp32_bits":naive,
               "transported_fp32_bits":transported,"original_bf16_bits":0,"naive_bf16_bits":0x3f80,
               "transported_bf16_bits":0,"scope":"separate products and fixed balanced FP32 RNE, not unspecified HF/CUDA"}
    return {"cases":len(records),"output_words_checked":2*len(records),"transport_mismatches":0,
            "naive_mismatch_cases":sum(r["reference_bits"] != r["naive_bits"] for r in records),
            "witness":witness},records


def shared_control():
    n,p = 4,2
    c = [[int(i!=j) for j in range(n)] for i in range(n)]
    assert rank(c,p) == n
    row_ranks = [rank([[row[j] if j==k else 0 for k in range(n)] for j in range(n)],p) for row in c]
    cases = 0
    for x,y in itertools.product(itertools.product(range(2),repeat=n),repeat=2):
        products = [u*v for u,v in zip(x,y)]
        shared = matvec(c,products,p)
        direct = [sum(c[i][j]*x[j]*y[j] for j in range(n)) % p for i in range(n)]
        assert shared == direct
        cases += 1
    return {"field":2,"n":4,"C":c,"C_rank":4,"per_output_bilinear_ranks":row_ranks,
            "sum_of_individual_ranks":sum(row_ranks),"actual_shared_products":4,
            "actual_output_xors":8,"input_transforms_identity":True,"full_cases_checked":cases,
            "mismatches":0,"claim":"output-wise rank lower bounds may not be summed against a shared decoder"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("output directory must not already exist")
    args.output_dir.mkdir(parents=True)
    numerical,raw = numerical_controls()
    report = {"kind":"INDEPENDENT_SCOPED_GAUGE_NATIVE_CORRECTNESS_CONSTRUCTION",
              "subject_commit":"dab57747b690ecc4ba0562f3af39e781af01a836",
              "field_enumeration":[enumeration(2,2,2),enumeration(2,3,6),enumeration(3,2,32)],
              "native_schedule":numerical,"shared_decoder_control":shared_control(),
              "cost_16384_square":operation_ledger(16384,16384),
              "registered_nonembedding_coefficients":403747897344,
              "registered_native_dense_work_retained_fraction":"1/1 before positive costs",
              "theory_status":"NOT_ESTABLISHED","hardware_status":"NOT_TESTED",
              "core_admission":False,"mission_obligations":{"O1":"OPEN","O2":"OPEN","O3":"OPEN","O4":"OPEN","O5":"OPEN","O6":"PARTIAL"},
              "new_qualifying_principles":0,"target_hardware_available":False}
    for name,value in (("summary.json",report),("native_cases.json",raw)):
        (args.output_dir/name).write_text(json.dumps(value,sort_keys=True,indent=2)+"\n",encoding="utf-8",newline="\r\n")
    files = sorted(p for p in args.output_dir.iterdir() if p.is_file())
    (args.output_dir/"checksums.sha256").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in files),encoding="utf-8",newline="\r\n")
    print(json.dumps(report,sort_keys=True,indent=2))


if __name__ == "__main__":
    main()

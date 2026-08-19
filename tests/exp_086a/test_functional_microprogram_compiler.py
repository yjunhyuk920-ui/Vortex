import copy
import pytest
import torch
from vortex_runtime.functional_microprogram_compiler import (
    MicroprogramError, cluster, exact_report, fit_library, residual_diagnostics, target_resources
)


def pairs(n=12):
    torch.manual_seed(4)
    x=torch.randn(n,6); w=torch.randn(6,5); b=torch.randn(5)
    return x.to(torch.bfloat16),(x@w+b).to(torch.bfloat16)


def test_cluster_deterministic():
    x,_=pairs(); a,c=cluster(x,3); a2,c2=cluster(x,3)
    assert torch.equal(a,a2) and torch.equal(c,c2) and set(a.tolist())=={0,1,2}


def test_library_deterministic_and_mutation_sensitive():
    x,y=pairs(); l,a,_=fit_library(x,y,3,6); l2,a2,_=fit_library(x,y,3,6)
    assert l.fingerprint==l2.fingerprint and torch.equal(a,a2)
    z=copy.deepcopy(y); z[0,0]=torch.nextafter(z[0,0],torch.tensor(float('inf'),dtype=torch.bfloat16))
    assert fit_library(x,z,3,6)[0].fingerprint!=l.fingerprint


def test_queries_and_fail_closed():
    x,y=pairs(); lib,_,_=fit_library(x,y,3,6)
    for oracle in (True,False):
        q=lib.query(x,y,'fp64' if oracle else 'fp32',oracle)
        assert q['candidate'].shape==y.shape and sum(q['use_counts'])==x.shape[0]
    with pytest.raises(MicroprogramError): lib.query(torch.full_like(x,float('nan')),y,'fp32',False)


def test_reports_and_resources():
    r=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.bfloat16); c=r.clone(); c[1,1]=4.5
    assert exact_report(r,c)['vector_exact_fraction']==.5
    assert residual_diagnostics(r,c)['unique_residual_vector_fraction']==1.0
    t=target_resources(4,32); assert t['sidecar_gib']<4 and t['compiled_mlp_operation_fraction']<.01
    assert t['whole_model_fraction_if_only_mlp_replaced']>.1


def test_full_rank_control_is_separate_from_low_rank_candidate():
    torch.manual_seed(17)
    x = torch.randn(20, 24, dtype=torch.bfloat16)
    y = torch.randn(20, 7, dtype=torch.bfloat16)
    candidate, assignments, centroids = fit_library(x, y, 2, 3)
    from vortex_runtime.functional_microprogram_compiler import fit_library_from_partition
    control = fit_library_from_partition(x, y, assignments, centroids, 20)
    assert control.query(x, y, 'fp64', False)['report']['vector_exact_fraction'] == 1.0
    assert candidate.query(x, y, 'fp64', False)['report']['vector_exact_fraction'] < 1.0

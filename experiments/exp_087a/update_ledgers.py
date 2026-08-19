from __future__ import annotations

import argparse,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];PREREG='<!-- EXP087A_CROSS_WEIGHT_CONTEXT_CODE -->';PREFIX='<!-- EXP087A_RESULT:'

def append_once(path:Path,marker:str,block:str)->None:
    current=path.read_text(encoding='utf-8') if path.exists() else ''
    if marker in current:return
    separator='' if not current or current.endswith('\n') else '\n'
    path.write_text(current+separator+'\n'+marker+'\n'+block.rstrip()+'\n',encoding='utf-8')

def preregister()->None:
    append_once(ROOT/'NEXT_EXPERIMENT.md',PREREG,"""## Active EXP-087A — Cross-Weight Context Dictionary Gate

The fixed BF16 significance-prefix source retained about 95% of exact MLP information. EXP-087A changes the source to aligned gate/up/down triplets, one- and two-source oracle tables, previous-triplet dictionaries, and procedural coordinate-tile generators. Every prediction is repaired by an exact XOR residual. The official SmolLM2 first/middle/last MLP layers are inspected with zero model forwards. Promotion requires the best favorable exact description fraction p50/p95 <=20%/25%.""")
    append_once(ROOT/'ASSUMPTION_REGISTER.md',PREREG,"""## EXP-087A frozen assumptions

- `A-087A-1`: aligned gate/up/down weights or spatial context may support a small exact generator plus low-entropy XOR residual. **UNVERIFIED**.
- `A-087A-2`: ideal entropy coding, free probability tables, free perfect hashing, and free decoder/kernel overhead make the Gate favorable. **FROZEN GRANTS**.
- `A-087A-3`: positive periodic controls prove the code family can cross 20% when the claimed structure exists. **CONTROL REQUIREMENT**.
- `A-087A-4`: rejection closes only the frozen aligned/local dictionary language, not every cross-layer nonlinear generator. **FIXED SCOPE**.""")
    append_once(ROOT/'VALIDATION_MATRIX.md',PREREG,"""## EXP-087A validation entry

| Gate | Required evidence | Before run |
|---|---|---|
| Official checkpoint | pinned SmolLM2 official loader | NOT RUN |
| Weight population | first/middle/last complete SwiGLU weights | FROZEN |
| Exact reconstruction | every predictor plus XOR residual reconstructs every word | NOT RUN |
| Positive control | periodic coordinate generator fraction <20% | UNIT TEST |
| Information Gate | best exact p50/p95 <=20%/25% | NOT RUN |
| Dense work / layer / hardware | separate later Gate | NOT TESTED |""")

def record(path:Path)->None:
    result=json.loads(path.read_text());source=str(result.get('source_sha','UNKNOWN'));marker=f'{PREFIX}{source} -->';decision=str(result.get('authoritative_decision','UNKNOWN'));a=result.get('DERIVED',{}).get('aggregate',{})
    block=f"""## EXP-087A result — `{source}`

```text
decision                         {decision}
layer count                      {a.get('layer_count','NOT_AVAILABLE')}
best information fraction p50   {a.get('best_information_fraction_p50','NOT_AVAILABLE')}
best information fraction p95   {a.get('best_information_fraction_p95','NOT_AVAILABLE')}
best candidate names             {a.get('best_candidate_names','NOT_AVAILABLE')}
all exact reconstructions        {a.get('all_reconstructions_exact','NOT_AVAILABLE')}
```

This is a favorable exact checkpoint-description Gate. Model forwards, dense-operation replacement, complete layer/state, 405B, 8 GiB, and physical latency remain `NOT TESTED`."""
    for name in ('RESEARCH_STATE.md','NEXT_EXPERIMENT.md','DECISION_LOG.md','FAILED_APPROACHES_RECENT.md','ASSUMPTION_REGISTER.md','VALIDATION_MATRIX.md'):append_once(ROOT/name,marker,block)
    latest=ROOT/'docs/research/EXP_087A_LATEST_RESULT.md';latest.parent.mkdir(parents=True,exist_ok=True);latest.write_text(marker+'\n'+block+'\n')

def main()->int:
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('preregister',));p.add_argument('--result',type=Path);a=p.parse_args()
    if a.mode=='preregister':preregister()
    elif a.result is not None:record(a.result)
    else:p.error('use --mode preregister or --result')
    return 0
if __name__=='__main__':raise SystemExit(main())

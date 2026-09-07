from pathlib import Path
import json
import numpy as np
from native import bf_bits,value,model
ROOT=Path(__file__).resolve().parents[1]
def run():
    lut=np.load(ROOT/'results/run/silu_abi.npy')
    G=U=bf_bits([[1.,1.]]);D=bf_bits([[1.]])
    y,_=model(bf_bits([[1.,1.],[1.,0.],[0.,1.]]),G,U,D,lut)
    stitched=bf_bits(np.add(value(y[1]),value(y[2]),dtype=np.float32))
    assert int(y[0,0])!=int(stitched[0])
    out={'whole_bits':int(y[0,0]),'stitched_bits':int(stitched[0]),
         'whole_value':float(value(y[0,0])),'stitched_value':float(value(stitched[0])),
         'partial_values':list(map(float,value(y[1:,0]))),
         'scope':'Naive additive composition of small independent input charts fails. No universal composition lower bound.',
         'status':'POST_MAIN_SCOPE_CHECK'}
    (ROOT/'results/scope.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(out)
if __name__=='__main__':run()

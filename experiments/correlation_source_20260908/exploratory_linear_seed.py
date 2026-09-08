# Pre-preregistration duplicate exploration, NOT a new result or core route.
# Preserved from the sandbox; existing E0 gate already excludes linear 2x3 seed.
from itertools import combinations
import json
for k in range(2,8):
    atoms=[1<<i for i in range(6)]+[(1<<(k-1))-1]
    covered=set(atoms)|{a^b for a,b in combinations(atoms,2)}|{0}
    subs=set()
    for a,b,c in combinations(sorted(covered-{0}),3):
        v=frozenset([0,a,b,c,a^b,a^c,b^c,a^b^c])
        if len(v)==8 and v<=covered:subs.add(v)
    print(k,len(covered),len(subs),flush=True)
    pairs=0; found=None
    for U in sorted(subs,key=lambda s: sorted(s)):
        ub=next((a,b,c) for a,b,c in combinations(sorted(U-{0}),3) if a^b!=c)
        for V in sorted(subs,key=lambda s: sorted(s)):
            if len(U&V)>1:continue
            pairs+=1
            from itertools import permutations
            for vb in permutations(sorted(V-{0}),3):
                if vb[0]^vb[1]==vb[2]:continue
                diag=[0]
                for a,b in zip(ub,vb):diag += [t^a^b for t in diag]
                if set(diag)<=covered:
                    found=(atoms,ub,vb);break
            if found:break
        if found:break
    print('pairs',pairs,'found',found,flush=True)

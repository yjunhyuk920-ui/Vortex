"""Sound restricted source rules. The exact-product proof is in docs/REPORT_KO.md.
Guards inspect constants, not response tables, and do not change the input domain.
"""
from fractions import Fraction
from rational import decode,encode,is_special
from engine import mul,fma

def exact_bf16_product(c):
    if c[0]!='const' or c[1]&65535:return False
    v=decode(c[1])
    if is_special(v) or abs(v)>1:return False
    if not v:return True
    s=abs(v.numerator);k=-(v.denominator.bit_length()-1)
    while s%2==0:s//=2;k+=1
    return k>=-16 and (s*255).bit_length()<=24

def rewrite(roots):
    memo={};proofs=[]
    def rec(t):
        if t in memo:return memo[t]
        if t[0] in ('input','const'):memo[t]=t;return t
        a=tuple([t[0]]+[rec(z) for z in t[1:]])
        if a[0]=='add':
            l,r=a[1:]
            if l[0]==r[0]=='mul' and l[2]==r[2] and l[2][0]=='input' and l[1][0]==r[1][0]=='const':
                av,bv=decode(l[1][1]),decode(r[1][1])
                if not is_special(av) and not is_special(bv) and av*bv>0:
                    cv=av+bv;cb=encode(cv);c=('const',cb)
                    if decode(cb)==cv and all(exact_bf16_product(z) for z in (l[1],r[1],c)):
                        proofs.append(dict(rule='same_input_exact_sum',constants=[l[1][1],r[1][1],cb]));a=mul(c,l[2])
            if a[0]=='add':
                l,r=a[1:]
                if r[0]=='mul' and r[2][0]=='input' and exact_bf16_product(r[1]):
                    proofs.append(dict(rule='exact_product_fma',constant=r[1][1]));a=fma(r[1],r[2],l)
        memo[t]=a;return a
    return [rec(t) for t in roots],proofs

def compile_matrix(weights):
    """Read a rectangular matrix of unmodified BF16 bit words; return both programs.
    This is the declared sequential FP32 reference ABI, not a HF/CUDA adapter.
    """
    from engine import add
    if not weights or not weights[0]:raise ValueError('empty matrix')
    m,n=len(weights),len(weights[0])
    if not (m<=128 and n<=128):raise ValueError('bounded interpreter dimensions')
    if any(len(row)!=n for row in weights):raise ValueError('ragged matrix')
    roots=[]
    for row in weights:
        if any(not isinstance(w,int) or not 0<=w<65536 for w in row):raise ValueError('invalid BF16 bits')
        ts=[mul(('const',w<<16),('input',j)) for j,w in enumerate(row)]
        s=ts[0]
        for t in ts[1:]:s=add(s,t)
        roots.append(s)
    candidate,proofs=rewrite(roots)
    return roots,candidate,proofs

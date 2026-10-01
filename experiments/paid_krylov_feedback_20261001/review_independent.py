import argparse,importlib.util,itertools,json,pathlib,random,subprocess,hashlib
p=pathlib.Path(__file__).with_name('verify.py')
spec=importlib.util.spec_from_file_location('v',p);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
parser=argparse.ArgumentParser();parser.add_argument('--out',type=pathlib.Path,required=True);out=parser.parse_args().out
v.main(out)
def mul(rows,s):
 return sum((sum(((row>>j)&1)*((s>>j)&1) for j in range(len(rows)))%2)<<i for i,row in enumerate(rows))
def par(c,s,n):return sum(((c>>j)&1)*((s>>j)&1) for j in range(n))%2
def independent_decode(z,K,n):
 return sum((sum(((col>>i)&1)*((z>>j)&1) for j,col in enumerate(K))%2)<<i for i in range(n))
# All sources A,b and observer triples for n<=2, every represented state and both inputs.
compiles=transitions=0;ranks={}
for n in (1,2):
 for rows in itertools.product(range(1<<n),repeat=n):
  for b in range(1<<n):
   for cs in itertools.product(range(1<<n),repeat=3):
    hot,dec,counts,meta=v.compile_source(v.source_bytes(rows,b,cs));nn,K=v.load_decoder(dec);r=len(K);compiles+=1;ranks[str((n,r))]=ranks.get(str((n,r)),0)+1
    for z in range(1<<r):
     s=independent_decode(z,K,n)
     for a in (0,1):
      e=v.Executor(hot,0xfedcba98);e.z=z
      y,rr,t,l0,l1,g=e.step(a)
      gg=par(cs[0],s,n)*par(cs[1],s,n);ss=mul(rows,s)^(b*(a^gg));yy=par(cs[2],ss,n)
      assert (independent_decode(e.z,K,n),y,g)==(ss,yy,gg)
      assert rr==(1664525*0xfedcba98+1013904223)%(1<<32)
      assert t==(rr<((1<<30)*(1+2*yy)))
      transitions+=1
# n=64 full-rank nilpotent companion exercises r=64, top-bit shift and mask serialization.
# max-sum rank-one case reaches all-ones and executes 64 products of two, sum=128.
cases=[('rank64',64,[0]+[1<<(i-1) for i in range(1,64)],1,[0xaaaaaaaaaaaaaaaa,0x5555555555555555,0x8000000000000001]),('maxsum64',64,[(1<<64)-1]*64,(1<<64)-1,[(1<<64)-1]*3)]
extra=[]
for name,n,rows,b,cs in cases:
 blob=v.source_bytes(rows,b,cs);hot,dec,counts,meta=v.compile_source(blob);_,K=v.load_decoder(dec)
 src=out/(name+'.source.bin');src.write_bytes(blob);inp=out/(name+'.inputs.bin');inputs=bytes([1]*256);inp.write_bytes(inputs)
 text=subprocess.check_output([str(out/'native_reference'),str(src),str(inp),str(0xffffffff),'256','0'],text=True)
 e=v.Executor(hot,0xffffffff)
 for i,line in enumerate(text.splitlines()):
  fields=line.split(',');ref=[int(x,16 if j in (2,6,7) else 10) for j,x in enumerate(fields)]
  y,rr,t,l0,l1,g=e.step(inputs[i]);s=independent_decode(e.z,K,n)
  assert [i,inputs[i],s,y,rr,t,l0,l1,g]==ref
 extra.append({'name':name,'n':n,'r':len(K),'steps':256,'pass':True})
# Source-only byte and trace replay agreement (summary contains output path differences).
orig=p.parent/'results';checked=0
for f in orig.iterdir():
 if f.suffix in ('.bin','.csv') or '.candidate' in f.name:
  assert f.read_bytes()==(out/f.name).read_bytes(),f.name;checked+=1
receipt={'status':'PASS','exhaustive_compilers':compiles,'exhaustive_state_input_transitions':transitions,'rank_histogram':ranks,'native_extra_cases':extra,'rerun_exact_science_files':checked,'reviewed_verify_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'reviewed_c_sha256':hashlib.sha256((p.parent/'native_reference.c').read_bytes()).hexdigest()}
(out/'independent_review.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

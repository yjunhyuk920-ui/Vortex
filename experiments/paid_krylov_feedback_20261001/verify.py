#!/usr/bin/env python3
"""Paid source compiler + independent candidate; no checkpoint/HF/GPU claim."""
import argparse, collections, hashlib, json, platform, random, struct, subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
HOT=struct.Struct('<4sIIQQQQQ')
def sha(b):return hashlib.sha256(b).hexdigest()
def pack_bf_bit(x):return 0x3f80 if x else 0

def source_bytes(rows,b,cs):
 n=len(rows); words=[]
 for row in rows:
  words.extend(0x4000 if (row>>j)&1 else 0x3f80 for j in range(n))
 words.extend(pack_bf_bit((b>>j)&1) for j in range(n))
 for c in cs:words.extend(0x4000 if (c>>j)&1 else 0x3f80 for j in range(n))
 return b'PKF1'+struct.pack('<I',n)+struct.pack('<'+'H'*len(words),*words)

def compile_source(blob):
 counts=collections.Counter();assert blob[:4]==b'PKF1';n=struct.unpack_from('<I',blob,4)[0];assert 1<=n<=64
 count=n*n+4*n;assert len(blob)==8+2*count;words=struct.unpack_from('<'+'H'*count,blob,8)
 counts['source_bf16_reads']=count;counts['source_bytes_read']=len(blob)
 rows=[];at=0
 for i in range(n):
  row=0
  for j in range(n):
   q=words[at];at+=1;assert q in (0x3f80,0x4000);row|=int(q==0x4000)<<j
  rows.append(row)
 b=0
 for j in range(n):
  q=words[at];at+=1;assert q in (0,0x3f80);b|=int(q!=0)<<j
 cs=[]
 for _ in range(3):
  c=0
  for j in range(n):
   q=words[at];at+=1;assert q in (0x3f80,0x4000);c|=int(q==0x4000)<<j
  cs.append(c)
 counts['source_pack_bit_steps']=count
 def mul(v):
  out=0;counts['matrix_calls']+=1
  for i,row in enumerate(rows):
   out|=((row&v).bit_count()&1)<<i
   counts['matrix_row_reads']+=1
  return out
 columns=[];pivots={};v=b
 while True:
  rem=v;coeff=0
  for p in range(n-1,-1,-1):
   counts['pivot_slots_checked']+=1
   if p in pivots and (rem>>p)&1:
    q,label=pivots[p];rem^=q;coeff^=label;counts['pivot_pair_reads']+=1;counts['pivot_xors']+=2
  if not rem:feedback=coeff;break
  j=len(columns);columns.append(v);pivots[rem.bit_length()-1]=(rem,coeff^(1<<j));counts['basis_column_writes']+=1;counts['pivot_pair_writes']+=1
  v=mul(v)
 r=len(columns);mask=(1<<r)-1;ds=[]
 for c in cs:
  d=0
  for j,v in enumerate(columns):
   d|=((c&v).bit_count()&1)<<j;counts['observer_column_reads']+=1;counts['observer_word_ops']+=5
  ds.append(d)
 counts['matrix_word_ops']=5*counts['matrix_row_reads']
 hot=HOT.pack(b'PKH1',n,r,feedback,*ds,mask)
 decode=b'PKD1'+struct.pack('<II',n,r)+b''.join(struct.pack('<Q',x) for x in columns)
 counts['hot_bytes_written']=len(hot);counts['decoder_bytes_written']=len(decode)
 # Post-construction certification is separate and paid, never free constructor work.
 validation_rows=0
 def raw_mul(v):
  nonlocal validation_rows
  validation_rows+=n
  return sum(((row&v).bit_count()&1)<<i for i,row in enumerate(rows))
 def dec(z):
  s=0
  for j,col in enumerate(columns):
   counts['postcheck_decoder_column_reads']+=1;counts['postcheck_decoder_bit_tests']+=1
   if (z>>j)&1:s^=col;counts['postcheck_decoder_xors']+=1
  return s
 for j,v in enumerate(columns):assert raw_mul(v)==dec((1<<(j+1)) if j+1<r else feedback)
 if r:assert columns[0]==b
 else:assert b==0
 for c,d in zip(cs,ds):
  for j,col in enumerate(columns):
   counts['postcheck_observer_column_reads']+=1;counts['postcheck_observer_word_ops']+=6
   assert ((c&col).bit_count()&1)==((d>>j)&1)
 counts['postcheck_matrix_row_reads']=validation_rows;counts['postcheck_matrix_word_ops']=5*validation_rows
 counts['postcheck_matrix_equality_tests']=r
 counts['postcheck_input_equality_tests']=1
 counts['fixed_array_build_word_instruction_upper']=64*(count+2*r*n+n*(r+1)+r*r+10*r+10*n+64)
 return hot,decode,dict(sorted(counts.items())),{'n':n,'r':r,'feedback_hex':hex(feedback),'transported_observers_hex':[hex(x) for x in ds]}

class Executor:
 def __init__(self,hot,rng):
  magic,self.n,self.r,self.f,self.d1,self.d2,self.do,self.mask=HOT.unpack(hot);assert magic==b'PKH1';assert 0<=self.r<=self.n<=64
  self.z=0;self.rng=rng
 def step(self,a):
  assert a in (0,1)
  g=((self.z&self.d1).bit_count()&1)&((self.z&self.d2).bit_count()&1)
  if self.r:
   top=(self.z>>(self.r-1))&1;self.z=((self.z<<1)&self.mask)^(self.f if top else 0)^(a^g)
  y=(self.z&self.do).bit_count()&1
  self.rng=(1664525*self.rng+1013904223)&0xffffffff
  token=int(self.rng<(0xc0000000 if y else 0x40000000))
  return y,self.rng,token,0x3f80 if y else 0,0 if y else 0x3f80,g

def load_decoder(blob):
 assert blob[:4]==b'PKD1';n,r=struct.unpack_from('<II',blob,4);assert len(blob)==12+8*r
 return n,list(struct.unpack_from('<'+'Q'*r,blob,12))
def decode_state(z,columns):
 s=0
 for j,v in enumerate(columns):
  if (z>>j)&1:s^=v
 return s

def main(out):
 out.mkdir(parents=True,exist_ok=True);rng=random.Random(20261001);cases=[]
 for name,n,kind in [('random32',32,'random'),('random64',64,'random'),('identity64',64,'identity'),('zero_input64',64,'zero_input')]:
  rows=[rng.getrandbits(n) for _ in range(n)];b=rng.getrandbits(n);cs=[rng.getrandbits(n) for _ in range(3)]
  if kind=='identity':rows=[1<<j for j in range(n)]
  if kind=='zero_input':b=0
  cases.append((name,source_bytes(rows,b,cs)))
 exe=out/'native_reference';compile_cmd=['cc','-std=c11','-O2','-ffp-contract=off',str(HERE/'native_reference.c'),'-o',str(exe)]
 subprocess.run(compile_cmd,check=True,capture_output=True)
 results=[];seed=0x12345678
 for name,blob in cases:
  src=out/(name+'.source.bin');src.write_bytes(blob);hot,decoder,counts,meta=compile_source(blob)
  (out/(name+'.hot.bin')).write_bytes(hot);(out/(name+'.decode.bin')).write_bytes(decoder)
  n,columns=load_decoder(decoder);r=len(columns)
  inputs=bytes(rng.randrange(2) for _ in range(512));inp=out/(name+'.inputs.bin');inp.write_bytes(inputs)
  runs=[]
  for mode in [0,1]:
   text=subprocess.check_output([str(exe),str(src),str(inp),str(seed),'512',str(mode)],text=True)
   (out/(name+f'.native{mode}.csv')).write_text(text)
   candidate=Executor((out/(name+'.hot.bin')).read_bytes(),seed);trace=[];previous_token=0;gate_ones=0;diff=0;readback_xors=0
   ablation=Executor(hot,seed);ablation.d1=0;ablation.d2=0
   for t,line in enumerate(text.splitlines()):
    fields=line.split(',');ref=[int(x,16 if i in (2,6,7) else 10) for i,x in enumerate(fields)]
    a=previous_token if mode else inputs[t];y,rr,token,l0,l1,g=candidate.step(a);s=decode_state(candidate.z,columns)
    got=[t,a,s,y,rr,token,l0,l1,g];assert got==ref,(name,mode,t,got,ref)
    previous_token=token;gate_ones+=g;ablation.step(a);diff+=int(ablation.z!=candidate.z);readback_xors+=candidate.z.bit_count()
    trace.append({'t':t,'input':a,'z':hex(candidate.z),'state':hex(s),'rng':rr,'token':token,'logits':[l0,l1],'feedback_gate':g})
   data=json.dumps(trace,sort_keys=True,separators=(',',':')).encode();(out/(name+f'.candidate{mode}.json')).write_bytes(data)
   runs.append({'mode':'own_output' if mode else 'external','steps':len(trace),'gate_ones':gate_ones,'ablation_differing_steps':diff,'trace_sha256':sha(data),'matches':True,'separate_readback_column_reads':len(trace)*r,'separate_readback_xors':readback_xors,'query_arithmetic_word_upper':32*len(trace),'fixed_array_query_all_word_instruction_upper':96*len(trace)})
  # The first forced impulse followed by zeros tests actual decoded-state corruption.
  mutation=None
  if r:
   fields=list(HOT.unpack(hot));fields[3]^=1;bad=Executor(HOT.pack(*fields),seed);good=Executor(hot,seed)
   for t in range(4*n+1):
    a=int(t==0);good.step(a);bad.step(a)
    if decode_state(good.z,columns)!=decode_state(bad.z,columns):mutation={'first_state_mismatch_step':t};break
  results.append({'name':name,**meta,'source_sha256':sha(blob),'source_bytes':len(blob),'hot_bytes':len(hot),'decoder_bytes':len(decoder),'counts':counts,'runs':runs,'corruption_witness':mutation,'native_products_per_step':n*n+3*n,'packed_dense_matrix_row_probes_per_step':n,'candidate_dense_matrix_probes_per_step':0,'separate_readback_basis_word_reads_per_step':r})
 # Exact host float32/BF16 witness, no torch or hidden kernel assumption.
 bits=struct.unpack('<I',struct.pack('<f',257.0))[0];bf=(bits+0x7fff+((bits>>16)&1))>>16;value=struct.unpack('<f',struct.pack('<I',bf<<16))[0];assert value==256.0
 summary={'status':'SCOPED_REFERENCE_PASS','full_mission_theory':'NOT_ESTABLISHED','hardware':'NOT_TESTED','core_admission':False,'cases':results,'rounding_witness':{'exact_sum':257,'bf16_word':hex(bf),'bf16_value':value},'compile_command':compile_cmd,'environment':{'python':platform.python_version(),'platform':platform.platform()},'validation_scope':'Declared native primitive only; no HF/GPU/latency/full-repository tests'}
 (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':summary['status'],'cases':[{k:x[k] for k in ['name','n','r','corruption_witness']} for x in results]},indent=2))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=HERE/'results');a=parser.parse_args();main(a.out)

/* Original supplied finite-word primitive. Not an HF model or sampler. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <assert.h>
static uint16_t get16(FILE*f){unsigned char b[2];assert(fread(b,1,2,f)==2);return b[0]|((uint16_t)b[1]<<8);}
static uint32_t get32(FILE*f){uint32_t v=0;for(int j=0;j<4;j++){int c=fgetc(f);assert(c!=EOF);v|=(uint32_t)c<<(8*j);}return v;}
static float frombf(uint16_t v){uint32_t w=(uint32_t)v<<16;float f;memcpy(&f,&w,4);return f;}
static uint16_t tobf(float f){uint32_t w;memcpy(&w,&f,4);w+=0x7fff+((w>>16)&1);return (uint16_t)(w>>16);}
static unsigned dot(const uint16_t*w,uint64_t s,unsigned n){
 float acc=0.0f; unsigned total=0;
 for(unsigned j=0;j<n;j++){float x=(float)((s>>j)&1);float prod=frombf(w[j])*x;acc=acc+prod;total+=(s>>j)&1;}
 float stored=frombf(tobf(acc)); assert(stored==acc);
 int val=(int)stored-(int)total;assert(val>=0 && val<=64);return (unsigned)val&1;
}
int main(int argc,char**argv){
 assert(argc==6); FILE*f=fopen(argv[1],"rb");assert(f);char magic[4];assert(fread(magic,1,4,f)==4&&!memcmp(magic,"PKF1",4));
 unsigned n=get32(f);assert(n>0&&n<=64);size_t count=(size_t)n*n+4*n;uint16_t*data=malloc(count*2);assert(data);
 for(size_t j=0;j<count;j++)data[j]=get16(f);assert(fgetc(f)==EOF);fclose(f);
 uint16_t*W=data,*b=data+n*n,*c1=b+n,*c2=c1+n,*co=c2+n;
 uint64_t bv=0;for(unsigned j=0;j<n;j++){assert(b[j]==0||b[j]==0x3f80);bv|=(uint64_t)(b[j]!=0)<<j;}
 uint32_t rng=(uint32_t)strtoul(argv[3],0,10);unsigned steps=(unsigned)strtoul(argv[4],0,10);int feedback=atoi(argv[5]);
 FILE*inputs=fopen(argv[2],"rb");assert(inputs);uint64_t s=0;unsigned token=0;
 for(unsigned t=0;t<steps;t++){
  int x=fgetc(inputs);assert(x==0||x==1);unsigned a=feedback?(t?token:0):(unsigned)x;
  unsigned g=dot(c1,s,n)&dot(c2,s,n);uint64_t next=0;
  for(unsigned i=0;i<n;i++)next|=(uint64_t)dot(W+i*n,s,n)<<i;
  if(a^g)next^=bv;s=next;unsigned y=dot(co,s,n);
  rng=1664525u*rng+1013904223u;uint32_t threshold=y?0xc0000000u:0x40000000u;token=rng<threshold;
  printf("%u,%u,%016" PRIx64 ",%u,%" PRIu32 ",%u,%04x,%04x,%u\n",t,a,s,y,rng,token,y?0x3f80:0,y?0:0x3f80,g);
 }
 fclose(inputs);free(data);return 0;
}

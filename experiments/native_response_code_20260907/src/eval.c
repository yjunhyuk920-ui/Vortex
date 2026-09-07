/* Loaded-file evaluator. No model weights, float operations, or activation table.
 * SHA-256 is verified by the Python loader before this independently parsed C path.
 * Format is little-endian; all length/shape/address guards are checked here as well. */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
static uint32_t u32(const uint8_t*p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static uint16_t u16(const uint8_t*p){return (uint16_t)p[0]|(uint16_t)p[1]<<8;}
int vx_query(const uint8_t*b,size_t len,const uint16_t*x,uint16_t*out){
 if(len<72||memcmp(b,"VXCODE01",8))return 1;
 uint32_t n=u32(b+8),q=u32(b+12),m=u32(b+16),L=u32(b+20),R=u32(b+24),r=u32(b+28),cb=u32(b+32),split=u32(b+36);
 if(!n||n>16||q<2||q>256||!m||m>4096||!split||split>=n||r>L||cb!=(r+7)/8)return 2;
 uint64_t lcheck=1,rcheck=1;
 for(uint32_t i=0;i<split;i++){lcheck*=q;if(lcheck>UINT32_MAX)return 2;}
 for(uint32_t i=split;i<n;i++){rcheck*=q;if(rcheck>UINT32_MAX)return 2;}
 if(lcheck!=L||rcheck!=R)return 2;
 uint64_t off=40+2*(uint64_t)q, data=off+(uint64_t)L*cb;
 if((uint64_t)len!=data+2*(uint64_t)R*r*m+32)return 3;
 /* Alphabet uniqueness and SHA-256 are validated once at load by Program.load. */
 uint32_t l=0,rr=0;
 for(uint32_t i=0;i<n;i++){
  uint32_t a=0;for(;a<q;a++)if(u16(b+40+2*a)==x[i])break;
  if(a==q)return 5;
  if(i<split)l=l*q+a;else rr=rr*q+a;
 }
 for(uint32_t j=0;j<m;j++)out[j]=0;
 const uint8_t*code=b+off+(uint64_t)l*cb;
 if(cb&&(r&7)&&(code[cb-1]>>(r&7)))return 6;
 for(uint32_t byte=0;byte<cb;byte++){
  uint32_t bits=code[byte];
  while(bits){
   uint32_t bit=(uint32_t)__builtin_ctz(bits),k=byte*8+bit; bits&=bits-1;
   if(k>=r)return 6;
   const uint8_t*p=b+data+2*((uint64_t)rr*r+k)*m;
   for(uint32_t j=0;j<m;j++)out[j]^=u16(p+2*j);
  }
 }
 return 0;
}

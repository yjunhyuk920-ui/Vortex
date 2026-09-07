#include <stdint.h>
#include <stdlib.h>
#include <string.h>
/* ABI: int16 coefficients and int8 ternary input; each row has a positive
 * coefficient; |sum of absolute products|<2^24. No FMA. Balanced FP32 sum. */
void ref(const int16_t* w,const int8_t* x,int m,int n,uint16_t* result){
 float* a=(float*)malloc((size_t)n*sizeof(float));
 if(!a) abort();
 for(int i=0;i<m;i++){
  for(int j=0;j<n;j++){volatile float p=(float)w[i*n+j]*(float)x[j];a[j]=p;}
  for(int len=n;len>1;len/=2)
   for(int j=0;j<len/2;j++){volatile float z=a[2*j]+a[2*j+1];a[j]=z;}
  uint32_t u;memcpy(&u,a,4);u+=0x7fff+((u>>16)&1);result[i]=(uint16_t)(u>>16);
 }
 free(a);
}

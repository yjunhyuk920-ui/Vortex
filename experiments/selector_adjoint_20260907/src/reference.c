/* Independent RNE conversion reference using exact double scaling, not the
   integer add-bias circuit. Compile without fast-math. Binary stdin: uint32;
   stdout: uint16. NaN payload/sign intentionally canonicalized by this ABI. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
static unsigned nearest_even(double a) {
    double q=floor(a), r=a-q;
    unsigned k=(unsigned)q;
    if(r>0.5 || (r==0.5 && (k&1)))++k;
    return k;
}
static uint16_t round16(uint32_t word) {
    float f;memcpy(&f,&word,4);
    unsigned sign=(word>>16)&0x8000;
    if(isnan(f))return 0x7fc0;
    if(isinf(f))return sign|0x7f80;
    double a=fabs((double)f);
    if(a==0)return sign;
    if(a<ldexp(1.0,-126))return sign|nearest_even(ldexp(a,133));
    int e;double m=frexp(a,&e);unsigned k=nearest_even(m*256.0);
    if(k==256){k=128;e++;}
    if(e>128)return sign|0x7f80;
    return (uint16_t)(sign|((unsigned)(e+126)<<7)|(k-128));
}
int main(void){uint32_t x;uint16_t y;while(fread(&x,4,1,stdin)==1){y=round16(x);if(fwrite(&y,2,1,stdout)!=1)return 2;}return ferror(stdin)?3:0;}

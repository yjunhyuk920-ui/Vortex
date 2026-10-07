#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#pragma STDC FENV_ACCESS ON

static uint32_t raw(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static void evaluate(float input, uint32_t *original, uint32_t *rewritten, int *original_flags, int *rewritten_flags) {
    volatile float x = input;
    feclearexcept(FE_ALL_EXCEPT);
    volatile float original_root = sqrtf(x);
    volatile float original_value = x / original_root;
    *original_flags = fetestexcept(FE_ALL_EXCEPT);
    *original = raw(original_value);
    feclearexcept(FE_ALL_EXCEPT);
    volatile float rewritten_value = sqrtf(x);
    *rewritten_flags = fetestexcept(FE_ALL_EXCEPT);
    *rewritten = raw(rewritten_value);
}

int main(void) {
    fenv_t caller;
    fenv_t held;
    uint32_t original3, rewritten3, original4, rewritten4;
    int original3_flags, rewritten3_flags, original4_flags, rewritten4_flags;
    if (fegetenv(&caller) || feholdexcept(&held) || fesetround(FE_TONEAREST)) return 2;
    evaluate(3.0f, &original3, &rewritten3, &original3_flags, &rewritten3_flags);
    evaluate(4.0f, &original4, &rewritten4, &original4_flags, &rewritten4_flags);
    if (fesetenv(&caller)) return 3;
    printf("{\"x3_original_bits\":\"0x%08x\",\"x3_rewritten_bits\":\"0x%08x\",\"x3_original_flags\":%d,\"x3_rewritten_flags\":%d,\"x4_original_bits\":\"0x%08x\",\"x4_rewritten_bits\":\"0x%08x\",\"x4_original_flags\":%d,\"x4_rewritten_flags\":%d,\"caller_fenv_restored\":true}\n", original3, rewritten3, original3_flags, rewritten3_flags, original4, rewritten4, original4_flags, rewritten4_flags);
    return !(original3 == 0x3fddb3d8u && rewritten3 == 0x3fddb3d7u && original4 == 0x40000000u && rewritten4 == 0x40000000u);
}

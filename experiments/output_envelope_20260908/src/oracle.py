"""Independent Fraction implementation of IEEE binary32 RNE for finite tests."""
from fractions import Fraction


def value(bits, fracbits=23, expbits=8, bias=127):
    sign = -1 if bits >> (fracbits+expbits) else 1
    e = (bits >> fracbits) & ((1 << expbits)-1)
    f = bits & ((1 << fracbits)-1)
    if e == (1 << expbits)-1:
        raise ValueError('nonfinite oracle input')
    if e == 0:
        mant, shift = f, 1-bias-fracbits
    else:
        mant, shift = (1 << fracbits)+f, e-bias-fracbits
    return sign * Fraction(mant) * (Fraction(2)**shift)


def rn(q: Fraction, fracbits=23, expbits=8, bias=127, negative_zero=False):
    sign = int(q < 0 or (q == 0 and negative_zero))
    q = abs(q)
    sbit = sign << (fracbits+expbits)
    if not q:
        return sbit
    e = q.numerator.bit_length() - q.denominator.bit_length()
    if q < Fraction(2)**e:
        e -= 1
    low = 1-bias
    shift = max(e, low)-fracbits
    scaled = q / Fraction(2)**shift
    a, rem = divmod(scaled.numerator, scaled.denominator)
    twice = rem*2
    a += int(twice > scaled.denominator or (twice == scaled.denominator and a % 2))
    if a == 0:
        return sbit
    if e < low and a < (1 << fracbits):
        return sbit | a
    e = max(e, low)
    if a >= (1 << (fracbits+1)):
        a >>= 1
        e += 1
    if e > bias:
        return sbit | (((1 << expbits)-1) << fracbits)
    return sbit | ((e+bias) << fracbits) | (a-(1 << fracbits))


def dot(w, x):
    states = []
    for wi, xi in zip(w, x):
        wi, xi = int(wi), int(xi)
        a, b = value(wi,7), value(xi,7)
        states.append(rn(a*b, negative_zero=bool((wi ^ xi) & 0x8000)))
    states += [0] * ((1 << (len(states)-1).bit_length())-len(states))
    while len(states) > 1:
        new = []
        for a,b in zip(states[::2],states[1::2]):
            q = value(a)+value(b)
            new.append(rn(q, negative_zero=(a==b==0x80000000)))
        states = new
    return rn(value(states[0]),7,negative_zero=(states[0]==0x80000000))

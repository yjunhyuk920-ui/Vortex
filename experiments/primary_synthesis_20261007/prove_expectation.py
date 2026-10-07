from fractions import Fraction
import json

def decode(bits):
    exp = (bits >> 23) & 255
    sig = (bits & ((1 << 23) - 1)) + (1 << 23)
    power = exp - 127 - 23
    return Fraction(sig) * Fraction(2) ** power

root_bits = 0x3fddb3d7
division_bits = 0x3fddb3d8
r = decode(root_bits)
lo_root = (decode(root_bits - 1) + r) / 2
hi_root = (r + decode(root_bits + 1)) / 2
sqrt_assertions = [lo_root * lo_root < 3, 3 < hi_root * hi_root]
q = Fraction(3) / r
q_lo = (decode(division_bits - 1) + decode(division_bits)) / 2
q_hi = (decode(division_bits) + decode(division_bits + 1)) / 2
division_assertions = [q_lo < q, q < q_hi]
assert all(sqrt_assertions + division_assertions)
print(json.dumps({
    "sqrt3_nearest_bits": hex(root_bits),
    "division_nearest_bits": hex(division_bits),
    "sqrt_midpoint_squared_proofs": sqrt_assertions,
    "division_midpoint_proofs": division_assertions,
    "ties": 0,
    "exact_fraction_proof": True,
    "models": 0,
    "gpu": 0,
}, indent=2))

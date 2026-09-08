import hashlib
import struct
import unittest
from fractions import Fraction
from source import Source, compile_source, direct, native_tree, ordered_tree, pack_ids
from costs import layout


def fraction_f32_integer(x: Fraction) -> Fraction:
    # Independent RN-even for the integer witness, not a general FP32 emulator.
    if x == 0:
        return Fraction(0)
    sign = -1 if x < 0 else 1
    a = abs(x)
    exponent = a.numerator.bit_length() - a.denominator.bit_length()
    if a < Fraction(2) ** exponent:
        exponent -= 1
    step = Fraction(2) ** max(0, exponent - 23)
    floor, rem = divmod(a, step)
    q = int(floor)
    if 2 * rem > step or (2 * rem == step and q & 1):
        q += 1
    return sign * q * step


class Tests(unittest.TestCase):
    def test_layout_matches_bytes(self):
        for k in range(1, 9):
            for n in range(1, 18):
                values = [i % (1 << k) for i in range(n)]
                self.assertEqual(len(compile_source(values, 1, n, k)),
                                 layout(n, k, len(set(values)))['conceptual_one_header_bytes'])

    def test_invalid_dimensions_and_functions(self):
        for shape in [(1.0, 2, 2), (1, 2, True), (2**32, 1, 2)]:
            with self.assertRaises(ValueError):
                compile_source([0, 1], *shape)
        s = Source(compile_source([0, 1], 1, 2, 2))
        with self.assertRaises(ValueError):
            s.parities(1, 3, [[4]])

    def test_all_two_site_tuples(self):
        # Every function of two bits, not only AND, on all source pairs/masks.
        for a in range(4):
            for b in range(4):
                src = Source(compile_source([a, b], 1, 2, 2))
                fs = [[m for m in range(4) if code >> m & 1] for code in range(16)]
                for mask in range(4):
                    got, _ = src.parities(1, mask, fs)
                    self.assertEqual(got, direct([a, b], 1, 2, 1, mask, fs))

    def test_missing_cross_information_repaired(self):
        xs = [Source(compile_source(p, 1, 2, 2)) for p in [[0, 3], [1, 2]]]
        ans = [s.parities(1, 3, [[1], [2], [3]])[0] for s in xs]
        self.assertEqual(ans, [[1, 1, 1], [1, 1, 0]])

    def test_constant_has_no_id_payload(self):
        src = Source(compile_source([3] * 16, 4, 4, 2))
        self.assertEqual(src.width, 0)
        ans, count = src.parities(15, 15, [[3]])
        self.assertEqual((ans, count.id_bytes), ([0], 0))
        self.assertEqual(count.histogram_updates, 16)

    def test_bit_packing(self):
        for width in range(1, 9):
            values = list(range(min(17, 1 << width)))
            s = Source(compile_source(values, 1, len(values), width))
            self.assertEqual([s.pattern(s.identifier(i)) for i in range(len(values))], values)

    def test_bad_checksum(self):
        data = bytearray(compile_source([0, 3], 1, 2, 2))
        data[32] ^= 1
        with self.assertRaises(ValueError):
            Source(bytes(data))

    def test_bad_length_with_valid_checksum(self):
        data = compile_source([0, 3], 1, 2, 2)
        body = data[:-33]
        with self.assertRaises(ValueError):
            Source(body + hashlib.sha256(body).digest())

    def test_bad_query(self):
        s = Source(compile_source([0, 1], 1, 2, 2))
        for args in [(-1, 1), (2, 1), (1, 4)]:
            with self.assertRaises(ValueError):
                s.counts(*args)

    def test_domain(self):
        for vals in [[4], [-1], [1.0], [True]]:
            with self.assertRaises(ValueError):
                compile_source(vals, 1, 1, 2)

    def test_native_order_witness(self):
        examples = [[2**24, 1, -2**24, 1], [2**24, -2**24, 1, 1]]
        observed = []
        for terms in examples:
            native = native_tree(terms)
            layer = list(map(Fraction, terms))
            while len(layer) > 1:
                layer = [fraction_f32_integer(layer[i] + layer[i + 1]) for i in range(0, len(layer), 2)]
            self.assertEqual(Fraction(native), layer[0])
            bits = [int.from_bytes(struct.pack('<f', v), 'little') for v in terms]
            source = Source(compile_source(bits, 1, 4, 32))
            got, cost = ordered_tree(source)
            self.assertEqual(got, native)
            self.assertEqual(cost['fp32_additions'], 3)
            observed.append((source.counts(1, 15)[0], got))
        self.assertEqual(observed[0][0], observed[1][0])
        self.assertEqual([x[1] for x in observed], [1.0, 2.0])


if __name__ == '__main__':
    unittest.main(verbosity=2)

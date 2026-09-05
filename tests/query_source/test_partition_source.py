import math
import struct
import unittest
from experiments.query_source.audit import RNG
from experiments.query_source.partition_source import build, query, reference, bits


class SourceTests(unittest.TestCase):
    def test_state_continuation(self):
        r = RNG(92455)
        W = [[bits(((r.next() % 257) - 128) / 64) for _ in range(8)] for _ in range(8)]
        idx = build(W)
        state, native_state = [0] * 8, [0] * 8
        for _ in range(16):
            x = [bits(((r.next() % 257) - 128) / 128) for _ in range(8)]
            state, cnt = query(idx, x, state)
            native_state = reference(W, x, native_state)
            self.assertEqual(state, native_state)
        # Accumulator continuation only, NOT Transformer KV or generation.

    def test_signed_zero_and_subnormal_words(self):
        W = [[0, 0x80000000, 1, 0x80000001],
             [0x80000000, 0, 0x80000001, 1]]
        idx = build(W)
        for x in [[0, 0x80000000, bits(1), bits(-1)], [bits(1)] * 4]:
            for initial in [[0, 0], [0x80000000, 0x80000000]]:
                self.assertEqual(query(idx, x, initial)[0], reference(W, x, initial))

    def test_rectangular_all_outputs(self):
        W = [[bits(i - j / 4) for j in range(5)] for i in range(3)]
        idx = build(W)
        x = [bits(0.25), bits(-1), bits(2), bits(0), bits(-0.0)]
        self.assertEqual(query(idx, x)[0], reference(W, x))

    def test_encoded_index_roundtrip(self):
        W = [[bits((i * 5 + j) % 7 / 8) for j in range(5)] for i in range(70)]
        idx = build(W)
        blob = idx.encode()
        magic, m, n, total, length = struct.unpack_from("<8sIIQQ", blob)
        self.assertEqual((magic, m, n, length), (b"QSPFMA01", 70, 5, len(blob)))
        at, seen, restored = 32, 0, [[None] * n for _ in range(m)]
        width = 8 * ((m + 63) // 64)
        for j in range(n):
            lo, hi, count = struct.unpack_from("<III", blob, at); at += 12
            membership = 0
            for _ in range(count):
                word = struct.unpack_from("<I", blob, at)[0]; at += 4
                mask = int.from_bytes(blob[at:at + width], "little"); at += width
                self.assertEqual(membership & mask, 0)
                membership |= mask
                for i in range(m):
                    if mask & (1 << i): restored[i][j] = word
                seen += 1
            self.assertEqual(membership, (1 << m) - 1)
        self.assertEqual((at, seen, restored), (len(blob), total, W))

    def test_validation_errors(self):
        for W in [[], [[]], [[0], [0, 0]], [[bits(math.inf)]]]:
            with self.assertRaises(ValueError): build(W)
        idx = build([[bits(1)]])
        with self.assertRaises(ValueError): query(idx, [])
        with self.assertRaises(ValueError): query(idx, [bits(math.nan)])


if __name__ == "__main__": unittest.main()

import hashlib
import struct
import unittest
from consumer_source import compile_consumers, ConsumerSource, check_direct

class ConsumerTests(unittest.TestCase):
    def test_all_binary_functions_two_sites(self):
        fs = [[m for m in range(4) if code >> m & 1] for code in range(16)]
        for a in range(4):
            for b in range(4):
                src = ConsumerSource(compile_consumers([a,b],2,fs)[0])
                for q in range(4):
                    self.assertEqual(src.query(q)[0],check_direct([a,b],fs,q))

    def test_all_boolean_functions_rank(self):
        fs = [[m for m in range(4) if code >> m & 1] for code in range(16)]
        _, cert = compile_consumers([0,1,2,3],2,fs)
        self.assertEqual(cert['rank'],3)

    def test_complements_share_plane(self):
        blob, cert = compile_consumers([0,1,2,3],2,[[3],[0,3]])
        self.assertEqual(cert['rank'],1)
        src = ConsumerSource(blob)
        for q in range(16):
            self.assertEqual(src.query(q)[0],check_direct([0,1,2,3],[[3],[0,3]],q))

    def test_constant_without_payload(self):
        blob, cert = compile_consumers([5]*17,3,[[0],[],[1,4]])
        self.assertEqual(cert['rank'],0)
        src = ConsumerSource(blob)
        for q in [0,1,31,(1<<17)-1]:
            self.assertEqual(src.query(q)[0],check_direct([5]*17,[[0],[],[1,4]],q))

    def test_cross_information(self):
        a = ConsumerSource(compile_consumers([0,3],2,[[3]])[0])
        b = ConsumerSource(compile_consumers([1,2],2,[[3]])[0])
        self.assertEqual([a.query(3)[0],b.query(3)[0]],[[1],[0]])

    def test_identity_consumers_require_rank(self):
        for k in range(1,9):
            w = [0]+[1<<i for i in range(k)]
            _, cert = compile_consumers(w,k,[[1<<i] for i in range(k)])
            self.assertEqual(cert['rank'],k)

    def test_rank_one_cannot_preserve_later_consumer(self):
        # Both AND values vanish but later observation of bit0 differs.
        a = ConsumerSource(compile_consumers([0],2,[[3]])[0])
        b = ConsumerSource(compile_consumers([1],2,[[3]])[0])
        self.assertEqual(a.query(1)[0],b.query(1)[0])
        self.assertNotEqual(check_direct([0],[[1]],1),check_direct([1],[[1]],1))

    def test_checksum_and_truncation(self):
        blob,_ = compile_consumers([0,1,2,3],2,[[3]])
        for bad in [blob[:-1],blob[:32]+bytes([blob[32]^1])+blob[33:]]:
            with self.assertRaises(ValueError):
                ConsumerSource(bad)

    def test_padding_rejected(self):
        blob,_ = compile_consumers([0,1,2],2,[[1]])
        body = bytearray(blob[:-32]); body[-1] |= 128
        with self.assertRaises(ValueError):
            ConsumerSource(bytes(body)+hashlib.sha256(body).digest())

    def test_domain(self):
        for words,k,fs in [([],2,[[1]]),([4],2,[[1]]),([True],2,[[1]]),([1],2,[[4]])]:
            with self.assertRaises(ValueError):
                compile_consumers(words,k,fs)
        src=ConsumerSource(compile_consumers([0,1],1,[[1]])[0])
        for q in [-1,4,True,1.0]:
            with self.assertRaises(ValueError):
                src.query(q)

    def test_cost_and_file_size(self):
        blob,cert = compile_consumers(list(range(16)),4,[[3],[1],[2],[4],[8]])
        src=ConsumerSource(blob)
        self.assertEqual(len(blob),56+src.f*(1+src.rb)+src.r*src.nb)
        _,cost=src.query((1<<16)-1)
        self.assertEqual(cost['source_plane_bytes'],src.r*src.nb)
        self.assertEqual(cost['plane_and_popcounts'],src.r)

if __name__=='__main__':
    unittest.main(verbosity=2)

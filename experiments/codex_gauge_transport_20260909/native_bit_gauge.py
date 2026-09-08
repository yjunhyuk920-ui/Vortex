from __future__ import annotations

import struct

from producer import bits, f32


def butterfly(words, width_bits=16, inverse=False):
    n = len(words)
    if n < 1 or n & (n-1):
        raise ValueError("word count must be a positive power of two")
    mask = (1 << width_bits)-1
    if any(v < 0 or v > mask for v in words):
        raise ValueError("word outside opaque bit width")
    output = list(words)
    stages = list(range(n.bit_length()-1))
    if inverse:
        stages.reverse()
    for stage in stages:
        half = 1 << stage
        for base in range(0,n,2*half):
            for offset in range(half):
                output[base+half+offset] ^= output[base+offset]
    return output


def bf16_multiply(a,b):
    x = struct.unpack("<f",struct.pack("<I",a << 16))[0]
    y = struct.unpack("<f",struct.pack("<I",b << 16))[0]
    word = bits(f32(x*y))
    if (word & 0x7f800000) == 0x7f800000 and word & 0x007fffff:
        return 0x7fc0
    return ((word + 0x7fff + ((word >> 16) & 1)) >> 16) & 0xffff


def encoded_multiply(z,w,native=bf16_multiply):
    if len(z) != len(w):
        raise ValueError("shape mismatch")
    x = butterfly(z,inverse=True)
    y = butterfly(w,inverse=True)
    output = [native(a,b) for a,b in zip(x,y)]
    return butterfly(output)


def cost(n,bits_per_word=16):
    if n < 1 or n & (n-1):
        raise ValueError("positive power of two required")
    stages = n.bit_length()-1
    xors = 3*n*stages//2
    return {
        "n":n,"bits_per_word":bits_per_word,"xor_word_operations":xors,
        "native_products":n,
        "xor_word_reads":2*xors,"xor_word_writes":xors,
        "internal_total_data_word_reads_including_copies_products":2*xors+5*n,
        "internal_total_data_word_writes_including_copies_products":xors+4*n,
        "algorithmic_live_word_upper_including_inputs":6*n,
        "packed_live_byte_upper_excluding_allocator_and_python_objects":6*n*((bits_per_word+7)//8),
        "costs_additional_to_original_gate":True,
        "arbitrary_dense_projection_implemented":False,
        "input_output_copy_and_index_work":"O(n log n), paid; not hardware timed",
        "native_callback_conversion_rounding_and_store":"paid n times",
        "hardware_allocator_KV_system_costs":"not measured and not granted free",
        "plain_ABI_adapter_additional_xors_for_two_input_encodes_one_output_decode":xors,
        "plain_ABI_adapter_total_xors":2*xors,
        "plain_ABI_adapter_live_word_upper_including_plain_inputs":8*n,
    }

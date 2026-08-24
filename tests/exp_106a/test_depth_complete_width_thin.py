from experiments.exp_106a.run_experiment import (
    D, bitplane_ledger, coordinate_control, flat_spectrum_control,
    linear_collision_control, q4_bytes, run, width_ledger,
)

def test_q4(): assert q4_bytes(128)==68 and q4_bytes(129)==73
def test_coordinate():
    x=coordinate_control(128,32,4,64); assert x["positive_exact"] and x["negative_immediate_mismatch"]
def test_dense_collision():
    x=linear_collision_control(); assert x["encoder_rank"]==16 and x["codes_byte_equal"] and x["target_distinguishes_collision"]
def test_flat(): assert flat_spectrum_control(64)["maximum_absolute_singular_error"]<1e-10
def test_small_width():
    x=width_ledger(128); assert x["fits_8gib"] and x["p50_scan_alone_pass"] and x["linear_kernel_dimension"]>0
def test_full_width():
    x=width_ledger(D); assert x["linear_kernel_dimension"]==0 and not x["fits_8gib"] and not x["p50_scan_alone_pass"]
def test_bitplane():
    x=bitplane_ledger(); assert not x["one_bitplane_fits_8gib"] and not x["one_bitplane_fits_p50_scan"]
def test_decision():
    x=run(); assert not x["integrity_failures"] and x["decision"].startswith("REJECT_DEPTH_COMPLETE")

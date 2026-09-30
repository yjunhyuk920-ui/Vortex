# Input-width/accounting refinement before validation 02

Validation 01 passed; its source, log and result are preserved unchanged.

Add explicit rejection of out-of-range format words and non-dyadic Fraction
addends. Record maximum supplied numerator/denominator widths so generic finite
dyadic inputs do not receive a free unit-cost arithmetic assumption. BF16 exact
products already obey fixed exponent/significand bounds. Rerun the same frozen
positive population and add only deterministic rejection checks for these newly
explicit API guards. No correctness or performance threshold is relaxed.

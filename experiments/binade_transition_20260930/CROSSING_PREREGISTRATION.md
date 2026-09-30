# Crossing extension preregistration

After the guarded-binade package, study a finite piecewise construction for normal,
positive/negative, changing dyadic-grid itineraries. This is another representation
result, never a coefficient/read reduction or backend authorization.

Hypothesis: for input grid h0 and fixed ordered uniform RNE-grid steps hj with exact
addends cj, let H=max(h0,hj). The composite is monotone, is equivariant under input
translation 2H, and after first reaching H has at most THREE output levels on one
half-open discrete input period. Three, not two, is required: a plateau may reach
the next period's endpoint before the period ends. A representation with <=2 cuts
and <=3 levels should admit exact constructive threshold inverses and subsequent
dyadic rounding steps without enumerating the period.

Before tests: compare explicitly iterated scalar grid rounding with this compact
map on all length-three sequences from grids {1/4,1,4} and addends
{-9/8,-1/2,0,1/2,9/8}, at input lattice integers -17..17. Check threshold inverses
for strict and non-strict comparisons and the translation law. Add FP32-scale
exponent-ratio witnesses requiring a period of 2^254; do not enumerate that period.

Separate the implemented itinerary map from a whole native chart constructor.
Any O(BN) chart bound needs monotone preimages, explicit cuts, equality/zero tags,
overflow handling and bit costs. All N products remain paid. If these obligations
are not implemented/checked, label the whole crossing constructor DERIVED only.

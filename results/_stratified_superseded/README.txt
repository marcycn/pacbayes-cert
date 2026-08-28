Superseded runs: label-stratified data partition.

These 100 runs used a class-stratified split of S into the prior subset S0 and
the bound subset S_b. Stratification was introduced to remove the prefix bias of
the original code, and it does, but it chooses the members of S0 from the class
counts of the whole of S, and S contains S_b. The prior therefore depended on the
labels of the bound set: disjoint index sets, but not the statistical
independence the PAC-Bayes prior condition requires. A sample with fixed class
quotas is also not an i.i.d. draw from D, which is what the concentration result
behind the bound assumes.

The partition is now a uniform random permutation, which removes the prefix bias
just as effectively and preserves both properties. Every learnt-prior cell was
re-run; these outputs are retained for audit only and are not cited anywhere.

Data-free-prior runs are unaffected: their split never had an S0 to choose, and
their split hashes reproduce byte-identically under the new code.

**If you want a painfree experience trying out new things in NumPy, this is likely not yet ready for you...**

This repository uses/exposes the new *experimental* NumPy
DType API.  NumPy has to be cutting edge to try (currently
this means working of a branch:
https://github.com/numpy/numpy/compare/main...seberg:experimental-dtype-api).

This is *very* early stage, so I will not take care about clean changesets.
Right now I expect to move this to the NumPy organization as soon
as the dust settles and things actually work.

I decided to use ``cython`` here, but I may change my mind or
do C and Cython depending for different things, the idea was that
it may be more useful.

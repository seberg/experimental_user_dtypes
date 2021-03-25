**If you want a painfree experience trying out new things in NumPy, this is likely not yet ready for you...**

This repository uses/exposes the new *experimental* NumPy
DType API.  NumPy has to be cutting edge to try (currently
this means working of a branch:
https://github.com/numpy/numpy/compare/main...seberg:experimental-dtype-api).

At this time (should change very soon), also use `NPY_USE_NEW_CASTINGIMPL=1`
as an environment variable during install to avoid certain issues.

This is *very* early stage, so I will not take care about clean changesets.
Right now I expect to move this to the NumPy organization as soon
as the dust settles and things actually work.

I decided to use ``cython`` here, but I may change my mind or
do C and Cython depending for different things, the idea was that
it may be more useful.


What is possible?

```python
from experimental_user_dtypes import float64unit as u, string_funcs; import numpy as np

F = np.array([u.Quantity(70., "Fahrenheit")])
C = F.astype(u.Float64UnitDType("Celsius"))
print(repr(C))
# array([21.11111111111115 °C], dtype='Float64UnitDType(degC)')
```

There is also a string comparison function in `string_funcs.string_equal` that works on
the NumPy bytes ("S" not "U") dtype.

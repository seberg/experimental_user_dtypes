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
import numpy as np
from experimental_user_dtypes import float64unit as u, string_funcs

F = np.array([u.Quantity(70., "Fahrenheit")])
C = F.astype(u.Float64UnitDType("Celsius"))
print(repr(C))
# array([21.11111111111115 °C], dtype='Float64UnitDType(degC)')

m = np.array([u.Quantity(5., "m")])
m_squared = m * m
print(repr(m_squared))
# array([25.0 m**2], dtype='Float64UnitDType(m**2)')

# If `string_funcs` is imported, this also works (i.e. `np.equal` with strings)
np.equal(np.array("string", dtype="S"), np.array("other_string", dtype="S"))
```
(Please don't multiple units that can't be multiply, it may crash and I have not checked
why yet.  The string equality only works on "S" not "U" dtype.)

As of now, only typical ufunc calls are included, reductions will _not_ work.
Note that certain options (such as providing an unaligned loop), will not yet
give any advantage for universal functions.

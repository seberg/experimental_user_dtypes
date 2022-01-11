**Crude examples using the new, experimental, DType API of NumPy**

This repository uses the new *experimental* NumPy DType and UFunc API.
Expect this to work on the NumPy main branch (although
at the time of writing this, it is not in yet, check the PR list).

This is an early stage, so I will not take care about clean changesets.
Right now I expect to move this to the NumPy organization as soon
as the dust settles and things actually work.

I decided to use ``cython`` here, but I may change my mind or
do C and Cython depending for different things, the idea was that
it may be more useful.

However, creating the correct DType is ugly in Cython, because as
of now it is only clean if creating a static type in C.
I hope to improve this, but it may require working with Python.

Creating a new ufunc loop is fine in cython though.

Please check the NumPy ``experimental_dtype_api.h`` header, the NEPs,
or ping me for more information.

What is possible?

Custom DType with ufunc and casting
-----------------------------------

```python
import numpy as np
from experimental_user_dtypes import float64unit as u

F = np.array([u.Quantity(70., "Fahrenheit")])
C = F.astype(u.Float64UnitDType("Celsius"))
print(repr(C))
# array([21.11111111111115 °C], dtype='Float64UnitDType(degC)')

m = np.array([u.Quantity(5., "m")])
m_squared = m * m
print(repr(m_squared))
# array([25.0 m**2], dtype='Float64UnitDType(m**2)')
```

(Please don't multiple units that can't be multiply, it may crash and I have not checked
why yet.)


Enhanced string equality
------------------------

```python
import numpy as np
from experimental_user_dtypes import string_funcs

# If `string_funcs` is imported, this also works (i.e. `np.equal` with strings)
np.equal(np.array(["string"], dtype="S"), np.array(["other_string"], dtype="S"))
# array([False])
```

Customizing NumPy Promotion
---------------------------

```python
import numpy as np
from experimental_user_dtypes import promoter_example

promoter_example.get_number_of_calls()
# 0

np.add([1., 2., 3.], [1, 2, 3])
# array([2., 4., 6.])

promoter_example.get_number_of_calls()
# 1
```
Shows that it is possible implement simple ufunc "promotion" based on the
common DType API.
(This modifies the ufunc! In this particular case the modification should
be completely compatible to current usage though.)

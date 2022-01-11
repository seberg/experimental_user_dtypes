from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF, Py_DECREF
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_Destructor

import numpy as np
cimport numpy as npc
from . cimport dtype_api


dtype_api.import_experimental_dtype_api(3)

"""
This module adds a generic promoter that is called whenver the first argument
is a double precision float to `np.add`  (exept in some cases due to value
based casting/promotion).

NumPy might reject such a promoter in the future.  The promoter here should
replicate (but generalize) the current NumPy behaviour.
"""
cdef int number_of_calls = 0


def get_number_of_calls():
    global number_of_calls
    return number_of_calls


cdef int simple_promoter(
        PyObject *ufunc,
        dtype_api.PyArray_DTypeMeta **op_dtypes,
        dtype_api.PyArray_DTypeMeta **signature,
        dtype_api.PyArray_DTypeMeta **new_dtypes) except -1:
    """The typical, simple promoter that maps to a homogeneous signature.

    Does not try to be particularly smart about the "signature":
    """
    # Note, should check that the ufunc has three operands
    global number_of_calls
    number_of_calls += 1

    cdef dtype_api.PyArray_DTypeMeta *dtypes[3]
    cdef npc.intp_t ndtypes = 0
    for i in range(3):
        if op_dtypes[i] != NULL:
            dtypes[i] = op_dtypes[i]
            ndtypes += 1

    cdef dtype_api.PyArray_DTypeMeta *common = dtype_api.PyArray_PromoteDTypeSequence(
            ndtypes, dtypes);

    for i in range(3):
        if signature[i] != NULL:
            Py_INCREF(<object>signature[i])
            new_dtypes[i] = signature[i]
        else:
            Py_INCREF(<object>common)
            new_dtypes[i] = common
    Py_DECREF(<object>common)
    return 0


cdef object capsule = PyCapsule_New(<void *>simple_promoter, "numpy._ufunc_promoter", <PyCapsule_Destructor>0)

dtype_api.PyUFunc_AddPromoter(np.add, (type(np.dtype(np.float64)), None, None), capsule)


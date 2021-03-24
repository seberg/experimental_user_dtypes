#cython: language_level=3

from cpython.object cimport PyObject, PyTypeObject
from . cimport dtype_api
cimport numpy as npc


dtype_api.import_experimental_dtype_api(0)


cdef class ScaledUInt8:
    cdef npc.npy_uint8 value
    cdef double min
    cdef double max


cdef object common_dtype(self, other):
    # Must return a DType (or NotImplemented)
    if other in (dtype_api.Float16, dtype_api.Float32, dtype_api.Float64):
        return other
    return NotImplemented


cdef object common_instance(descr1, descr2):
    raise ValueError("Not implemented, maybe will even force exact match?")


cdef object discover_descr_from_pyobject(cls, obj):
    # Must return a descriptor (or an error).
    assert(isinstance(obj, ScaledUInt8))
    cdef ScaledUInt8 scaled_uint8 = <ScaledUInt8>obj

    raise NotImplementedError("Should use the min/max to create a range here!")


class ScaledUInt8DTypeBase:
    def __new__(cls):
        return 123

    def range(self):
        return 3.1415


cdef dtype_api.PyArrayDTypeMeta_Spec spec
spec.name = "ScaledUInt8DType"
spec.typeobj = <PyTypeObject *>ScaledUInt8
spec.flags = dtype_api.NPY_DTYPE_PARAMETRIC
spec.slots = NULL
spec.baseclass = <PyTypeObject *>ScaledUInt8DTypeBase

ScaledUInt8DType = dtype_api.PyArrayDTypeMeta_FromSpec(&spec)


#cython: language_level=3

from cpython.object cimport (
        PyObject, PyTypeObject, Py_TPFLAGS_DEFAULT, Py_TPFLAGS_BASETYPE)
from cpython.ref cimport Py_INCREF
from cpython.type cimport PyType_Ready
from libc.string cimport memcpy
cimport numpy as npc

from . cimport dtype_api
from . import dtypes as _npdtypes


# Using unyt, with no particular bias. Just happened to check it first this
# time around:
from unyt import Unit
import numpy as np


__all__ = ["Quantity", "Float64UnitDType"]

dtype_api.import_experimental_dtype_api(3)


cdef class Quantity:
    cdef readonly double value
    cdef readonly object unit

    def __cinit__(self, value, unit):
        self.value = value
        self.unit = Unit(unit)

    def __repr__(self):
        return f"{self.value} {self.unit}"


cdef object common_dtype(self, other):
    # Must return a DType (or NotImplemented)
    if other in (_npdtypes.Float16, _npdtypes.Float32, _npdtypes.Float64):
        return other
    return NotImplemented


cdef object common_instance(descr1, descr2):
    raise ValueError("Not implemented, maybe will even force exact match?")


cdef object discover_descr_from_pyobject(cls, obj):
    cdef Quantity quantity
    # Must return a descriptor (or an error).
    if isinstance(obj, Quantity):
        quantity = <Quantity>obj
        return cls(quantity.unit)

    # Must be a builtin, use "dimensionless":
    return cls("")


cdef int setitem(_Float64UnitDTypeBase self, object obj, char *ptr) except -1:
    cdef double value

    if isinstance(obj, Quantity):
        value = obj.value
        if obj.unit != self.unit:
            raise NotImplementedError("Scalar assignment with unit conversion not implemented yet")
    else:
        if not self.unit.is_dimensionless:
            raise ValueError("Can only assign value to a dimensionless array.")
        # use cythons conversion to float (which will convert Python builtins)
        value = obj

    # This allows force-casting dimensionless to dimension.
    memcpy(ptr, <void *>&value, sizeof(double))
    return 0


cdef object getitem(_Float64UnitDTypeBase self, char *ptr):
    cdef double value
    memcpy(<void *>&value, ptr, sizeof(double))
    return Quantity(value, self.unit)


cdef class _Float64UnitDTypeBase(npc.dtype):
    # This is an aweful hack, this type must NOT be used!
    # Further, this will lock us in on the ABI size of `PyArray_Descr`
    # (which is probably fine, lets just add a `void *reserved` maybe?)
    cdef readonly object unit
    def __cinit__(self, unit):
        if (type(type(self)) is not type(np.dtype)
                or type(self) is _Float64UnitDTypeBase):
            # The "base" class must not be instantiated (the first check should
            # cover the second one as well really).
            raise RuntimeError("Invalid use of implementation detail!")
        self.unit = Unit(unit)
        self.itemsize = sizeof(double)
        self.alignment = sizeof(double)

    @property
    def name(self):
        return f"Float64UnitDType({self.unit!r})"

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def si(self):
        """Returns the SI verison of the dtype"""
        # I assume `get_base_equivalent()` gives SI?
        return Float64UnitDType(self.unit.get_base_equivalent())


cdef npc.NPY_CASTING unit_to_unit_cast_resolve_descriptors(method,
        PyObject *dtypes[2],
        PyObject *descrs[2], PyObject *descrs_out[2],
        npc.intp_t *view_offset) except <npc.NPY_CASTING>-1:
    """
    The resolver function, could possibly provide a default for this type
    of unary/casting resolver.
    """
    cdef npc.NPY_CASTING casting
    if descrs[1] == <PyObject *>0:
        # Shouldn't really happen a cast function?
        Py_INCREF(<object>descrs[0])
        descrs_out[0] = <PyObject *>descrs[0]
        Py_INCREF(<object>descrs[0])
        descrs_out[1] = <PyObject *>descrs[0]

    # For units, we have to check it "twice" unfortunately (currently more
    # potentially):
    conv = (<_Float64UnitDTypeBase>descrs[0]).unit.get_conversion_factor(
                (<_Float64UnitDTypeBase>descrs[1]).unit)
    if conv == (1.0, None):
        casting = npc.NPY_SAFE_CASTING
        view_offset[0] = 0  # cast can be represented by a view, indicate this
    else:
        # Casting can be lossy, so don't lie about that:
        casting = npc.NPY_SAME_KIND_CASTING

    Py_INCREF(<object>descrs[0])
    descrs_out[0] = <PyObject *>descrs[0]
    Py_INCREF(<object>descrs[1])
    descrs_out[1] = <PyObject *>descrs[1]
    return casting


cdef int string_equal_strided_loop(
        dtype_api.PyArrayMethod_Context *context,
        char **data, npc.intp_t *dimensions, npc.intp_t *strides,
        void *userdata) nogil except -1:
    cdef npc.intp_t N = dimensions[0]
    cdef char *vals_in = <char *>data[0]
    cdef char *vals_out = <char *>data[1]
    cdef npc.intp_t strides_in = strides[0]
    cdef npc.intp_t strides_out = strides[1]

    cdef double factor
    cdef double offset = 0
    with gil:
        # NOTE: This GIL block should be part of of the _get_loop, but I
        #       did not make that publically available yet due to bad API.
        #       We could at least "cache" these in `userdata`, but that would
        #       require working with the NpyAuxdata, so lets not do that...
        unit1 = (<_Float64UnitDTypeBase>context.descriptors[0]).unit
        unit2 = (<_Float64UnitDTypeBase>context.descriptors[1]).unit

        factor, offset_obj = unit1.get_conversion_factor(unit2)
        if offset_obj is not None:
            offset = offset_obj

    for i in range(N):
        (<double *>vals_out)[0] = (<double *>vals_in)[0] * factor - offset

        vals_in += strides_in
        vals_out += strides_out

    return 0


cdef int string_equal_strided_loop_unaligned(
        dtype_api.PyArrayMethod_Context *context,
        char **data, npc.intp_t *dimensions, npc.intp_t *strides,
        void *userdata) nogil except -1:
    cdef npc.intp_t N = dimensions[0]
    cdef char *vals_in = data[0]
    cdef char *vals_out = data[1]
    cdef npc.intp_t strides_in = strides[0]
    cdef npc.intp_t strides_out = strides[1]

    cdef double value
    cdef double factor
    cdef double offset = 0

    with gil:
        unit1 = (<_Float64UnitDTypeBase>context.descriptors[0]).unit
        unit2 = (<_Float64UnitDTypeBase>context.descriptors[1]).unit

        factor, offset_obj = unit1.get_conversion_factor(unit2)
        if offset_obj is not None:
            offset = offset_obj

    for i in range(N):
        memcpy(&value, vals_in, sizeof(double))
        value = value * factor + offset
        memcpy(vals_out, &value, sizeof(double))
        vals_in += strides_in
        vals_out += strides_out

    return 0



cdef dtype_api.PyArrayDTypeMeta_Spec spec
spec.typeobj = <PyTypeObject *>Quantity
spec.flags = dtype_api.NPY_DT_PARAMETRIC

# Generic DType slots:
cdef dtype_api.PyType_Slot slots[5]
spec.slots = slots
slots[0].slot = dtype_api.NPY_DT_common_dtype
slots[0].pfunc = <void *>common_dtype
slots[1].slot = dtype_api.NPY_DT_common_instance
slots[1].pfunc = <void *>common_instance
slots[2].slot = dtype_api.NPY_DT_setitem
slots[2].pfunc = <void *>setitem
slots[3].slot = dtype_api.NPY_DT_getitem
slots[3].pfunc = <void *>getitem
slots[4].slot = dtype_api.NPY_DT_discover_descr_from_pyobject
slots[4].pfunc = <void *>discover_descr_from_pyobject
# Sentinel:
slots[5].slot = 0
slots[5].pfunc = <void *>0


# Define all casts::
cdef dtype_api.PyArrayMethod_Spec *castingimpls[2]
spec.casts = &castingimpls[0]

# First cast (from one unit to another "within the same DType")
cdef dtype_api.PyArrayMethod_Spec unit_to_unit_cast_spec
castingimpls[0] = &unit_to_unit_cast_spec

unit_to_unit_cast_spec.name = "unit_to_unit_cast"
unit_to_unit_cast_spec.nin = 1
unit_to_unit_cast_spec.nout = 1
# We have to get the GIL briefly currently. Note that floating point checks
# currently do not happen for casts, this is a NumPy bug:
unit_to_unit_cast_spec.flags = dtype_api.NPY_METH_SUPPORTS_UNALIGNED

cdef PyObject *dtypes[2]
unit_to_unit_cast_spec.dtypes = dtypes
# We don't know the new DType yet, so use NULL:
dtypes[0] = <PyObject *>0
dtypes[1] = <PyObject *>0

cdef dtype_api.PyType_Slot meth_slots[4]
unit_to_unit_cast_spec.slots = meth_slots
meth_slots[0].slot = dtype_api.NPY_METH_resolve_descriptors
meth_slots[0].pfunc = <void *>unit_to_unit_cast_resolve_descriptors
meth_slots[1].slot = dtype_api.NPY_METH_strided_loop
meth_slots[1].pfunc = <void *>string_equal_strided_loop
meth_slots[2].slot = dtype_api.NPY_METH_unaligned_strided_loop
meth_slots[2].pfunc = <void *>string_equal_strided_loop_unaligned
# End of casts sentinel:
castingimpls[1] = <dtype_api.PyArrayMethod_Spec *>0


spec.baseclass = <PyTypeObject *>_Float64UnitDTypeBase


# Use the C API to create the actual static type, so that we can give
# it the size of PyArray_DTypeMeta.  This is very ugly, and it would likely
# be better to just not do it in cython.  But I started in Cython, and it seems
# easier to do it "manually" here, rather than moving everything to C, or
# making a dance to define functions in Cython and then export to C.
cdef dtype_api.PyArray_DTypeMeta float64_struct

# TODO: Should use proper initialization marcos!
# TODO: is there a better way to write this even if we embrace the approach?
cdef PyObject *float64_as_obj = <PyObject *>&float64_struct
cdef type _dtypemeta = <type>&dtype_api.PyArrayDTypeMeta_Type
float64_as_obj[0].ob_type = <PyTypeObject *>_dtypemeta
float64_as_obj[0].ob_refcnt = 1

cdef PyTypeObject *float64_as_type = <PyTypeObject *>&float64_struct
float64_as_type[0].tp_name = "experimental_user_dtypes.float64unit.Float64DType"
float64_as_type[0].tp_base = <PyTypeObject *>_Float64UnitDTypeBase
float64_as_type[0].tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE

PyType_Ready(<object>&float64_struct)
dtype_api.PyArrayInitDTypeMeta_FromSpec(
        <dtype_api.PyArray_DTypeMeta *>&float64_struct, &spec)

Float64UnitDType = <object>&float64_struct

#
# Puh, we got a DType, now lets create a multiply function :)
#
# NOTE: Please be aware that this is NOT how I imagine Units to work!
#       Units should be able to re-use existing loops, so hopefully it would
#       only have to define something similar to the resolve_descriptors
#       but "inherit" the rest (See the NEP 43 draft)
#
# (One more note, we could reuse the above specs/slots, NumPy is done with them)


cdef npc.NPY_CASTING multiply_units_resolve_descriptors(method,
        PyObject *dtypes[3],
        PyObject *descrs[3], PyObject *descrs_out[3],
        npc.intp_t *view_offset) except <npc.NPY_CASTING>-1:
    """
    The resolver function, could possibly provide a default for this type
    of unary/casting resolver.
    """
    cdef npc.NPY_CASTING casting
    if descrs[1] == <PyObject *>0:
        # Shouldn't really happen a cast function?
        Py_INCREF(<object>descrs[0])
        descrs_out[0] = <PyObject *>descrs[0]
        Py_INCREF(<object>descrs[0])
        descrs_out[1] = <PyObject *>descrs[0]

    try:
        out_unit = ((<_Float64UnitDTypeBase>descrs[0]).unit *
                    (<_Float64UnitDTypeBase>descrs[1]).unit)
    except:
        # Clearning the error should cause a generic error to be given.
        # Note that the return value is a detail I still am thinking about
        # changing.
        return <npc.NPY_CASTING>-1

    if (descrs[2] != <PyObject *>0
            and (<_Float64UnitDTypeBase>descrs[2]).unit == out_unit):
        # retain exact instance when passed in, currently necessary for things
        # to work smoothly, but should only be a mild performance improvement.
        out_dtype = <object>descrs[2]
    else:
        out_dtype = Float64UnitDType(out_unit)

    Py_INCREF(<object>descrs[0])
    descrs_out[0] = <PyObject *>descrs[0]
    Py_INCREF(<object>descrs[1])
    descrs_out[1] = <PyObject *>descrs[1]
    Py_INCREF(out_dtype)
    descrs_out[2] = <PyObject *>out_dtype
    return npc.NPY_SAFE_CASTING


cdef int multiply_units_strided_loop(
        dtype_api.PyArrayMethod_Context *context,
        char **data, npc.intp_t *dimensions, npc.intp_t *strides,
        void *userdata) nogil:
    cdef npc.intp_t N = dimensions[0]
    cdef char *vals_in0 = <char *>data[0]
    cdef char *vals_in1 = <char *>data[1]
    cdef char *vals_out = <char *>data[2]
    cdef npc.intp_t strides_in0 = strides[0]
    cdef npc.intp_t strides_in1 = strides[1]
    cdef npc.intp_t strides_out = strides[2]

    for i in range(N):
        (<double *>vals_out)[0] = (<double *>vals_in0)[0] * (<double *>vals_in1)[0]
        vals_in0 += strides_in0
        vals_in1 += strides_in1
        vals_out += strides_out


# We now declare the `spec` and fill it (it can be discarted later).
cdef dtype_api.PyArrayMethod_Spec multiply_spec

# Basic information:
multiply_spec.name = "unit_multiply"
multiply_spec.nin = 2
multiply_spec.nout = 1

# Define the dtypes we operate on:
cdef PyObject *multiply_dtypes[3]
multiply_spec.dtypes = multiply_dtypes
multiply_dtypes[0] = <PyObject *>Float64UnitDType
multiply_dtypes[1] = <PyObject *>Float64UnitDType
multiply_dtypes[2] = <PyObject *>Float64UnitDType


# Define the function:
cdef dtype_api.PyType_Slot multiply_slots[3]
multiply_spec.slots = multiply_slots

# Pass the function using the 0 terminated slots.
multiply_slots[0].slot = dtype_api.NPY_METH_resolve_descriptors
multiply_slots[0].pfunc = <void *>multiply_units_resolve_descriptors
multiply_slots[1].slot = dtype_api.NPY_METH_strided_loop
multiply_slots[1].pfunc = <void *>multiply_units_strided_loop
multiply_slots[2].slot = 0
multiply_slots[2].pfunc = <void *>0

# Not used right now, but we can indicate not to check float errors,
# we do not require the GIL to be held, which is the default (currently):
multiply_spec.flags = 0


dtype_api.PyUFunc_AddLoopFromSpec(np.multiply, &multiply_spec)


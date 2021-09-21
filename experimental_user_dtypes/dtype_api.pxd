#cython: language_level=3

from cpython.object cimport PyObject, PyTypeObject
cimport numpy as npc


cdef extern from "Python.h":
    ctypedef struct PyType_Slot:
        int slot
        void *pfunc


cdef extern from "numpy/experimental_dtype_api.h":
    int import_experimental_dtype_api(int version) except -1

    cdef enum NPY_ARRAYMETHOD_FLAGS:
        NPY_METH_REQUIRES_PYAPI
        NPY_METH_NO_FLOATINGPOINT_ERRORS
        NPY_METH_SUPPORTS_UNALIGNED

    ctypedef struct PyArrayMethod_Spec:
        const char *name
        int nin, nout
        npc.NPY_CASTING casting
        # TODO: flags should be declared `NPY_ARRAYMETHOD_FLAGS`
        #       (although maybe it makes sense to ensure it doesn't matter)
        int flags
        PyObject **dtypes
        PyType_Slot *slots

    ctypedef struct PyArrayMethod_Context:
        PyObject *caller  # Caller (would be the ufunc, but may be NULL.)
        PyObject *method  # The method "self". Currently an opaque object
        # The descriptors:
        npc.PyArray_Descr **descriptors

    int NPY_METH_resolve_descriptors
    int NPY_METH_strided_loop
    int NPY_METH_contiguous_loop
    int NPY_METH_unaligned_strided_loop
    int NPY_METH_unaligned_contiguous_loop

    ctypedef int PyArrayMethod_StridedLoop(PyArrayMethod_Context *context,
        char **data, npc.intp_t *dimensions, npc.intp_t *strides,
        void *transferdata) except -1

    int PyUFunc_AddLoopFromSpec(object ufunc, PyArrayMethod_Spec *spec) except -1

    #
    # DType API.
    #
    int NPY_DT_PARAMETRIC
    int NPY_DT_ABSTRACT

    int NPY_DT_discover_descr_from_pyobject
    int _NPY_DT_is_known_scalar_type
    int NPY_DT_default_descr
    int NPY_DT_common_dtype
    int NPY_DT_common_instance
    int NPY_DT_setitem
    int NPY_DT_getitem

    ctypedef struct PyArray_DTypeMeta:
        # We don't really know what is inside (just its size in C)
        pass

    ctypedef struct PyArrayDTypeMeta_Spec:
        char *name
        PyTypeObject *typeobj
        int flags
        PyArrayMethod_Spec **casts
        PyType_Slot *slots
        PyTypeObject *baseclass

    int PyArrayInitDTypeMeta_FromSpec(
            PyArray_DTypeMeta *DType, PyArrayDTypeMeta_Spec *spec) except -1

    # Not exported in the normal NumPy pxd (should be part of the enum)
    cdef npc.NPY_CASTING NPY_CAST_IS_VIEW = <npc.NPY_CASTING>(1 << 16)


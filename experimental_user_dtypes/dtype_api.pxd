#cython: language_level=3

from cpython.object cimport PyObject, PyTypeObject
cimport numpy as npc


cdef extern from "Python.h":
    struct PyType_Slot:
        int slot
        void *pfunc


cdef extern from "numpy/experimental_dtype_api.h":
    int import_experimental_dtype_api(int version) except -1

    enum NPY_ARRAYMETHOD_FLAGS:
        NPY_METH_REQUIRES_PYAPI
        NPY_METH_NO_FLOATINGPOINT_ERRORS
        NPY_METH_SUPPORTS_UNALIGNED

    ctypedef struct PyArrayMethod_Spec:
        const char *name
        int nin, nout
        npc.NPY_CASTING casting
        NPY_ARRAYMETHOD_FLAGS flags
        PyObject **dtypes
        PyType_Slot *slots

    ctypedef struct PyArrayMethod_Context:
        PyObject *caller  # Caller (would be the ufunc, but may be NULL.)
        PyObject *method  # The method "self". Currently an opaque object
        # The descriptors:
        npc.PyArray_Descr **descriptors


    int NPY_METH_strided_loop
    int NPY_METH_contiguous_loop
    int NPY_METH_unaligned_strided_loop
    int NPY_METH_unaligned_contiguous_loop

    ctypedef int PyArrayMethod_StridedLoop(PyArrayMethod_Context *context,
        char **data, npc.intp_t *dimensions, npc.intp_t *strides,
        void *transferdata) except -1

    object PyArrayMethod_FromSpec(PyArrayMethod_Spec *spec)


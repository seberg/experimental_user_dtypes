#cython: language_level=3

from cpython.object cimport PyObject

import numpy as np
cimport numpy as npc
from . cimport dtype_api
from libc.string cimport memcmp


"""
The ArrayMethod will be used also for registering ufuncs.  At this time
this is not possible. But it is possible to use (private) methods to
create future UFunc-likes.

The public API does not allow to optimize the loop all the way, but
we will only write a strided inner-loop right now.
"""


cdef int string_equality(dtype_api.PyArrayMethod_Context *context,
        char **data, npc.intp_t *dimensions, npc.intp_t *strides,
        void *userdata):  # should be nogil...
    """
    The data field is currently not accessible, since it would require
    either an additional "setup" function or exposure of `get_loop`.
    It could be used if set to an NpyAuxdata...
    context passes in some additional information. For users most
    importantly the descriptors at this time.
    """
    cdef npc.intp_t N = dimensions[0]
    cdef int elsize0 = (<npc.dtype>context.descriptors[0]).elsize
    cdef int elsize1 = (<npc.dtype>context.descriptors[1]).elsize

    cdef int len_long, len_short
    cdef npc.intp_t stride_long, stride_short

    cdef char *in_long
    cdef char *in_short
    cdef char *out = data[2]

    if  elsize0 >= elsize1:
        in_long = data[0]
        in_short = data[1]
        len_long = elsize0
        len_short = elsize1
        stride_long = strides[0]
        stride_short = strides[1]
    else:
        in_long = data[1]
        in_short = data[0]
        len_long = elsize1
        len_short = elsize0
        stride_long = strides[1]
        stride_short = strides[0]

    if len_long == len_short:
        # If they are the same length, just do a straight comparison:
        for i in range(N):
            if memcmp(in_long, in_short, len_long) == 0:
                out[i] = 1
            else:
                out[i] = 0

            in_long += stride_long
            in_short += stride_short
        return 0  # always succeeds (always 0)

    cdef int additional = len_long - len_short
    for i in range(N):
        if not memcmp(in_long, in_short, len_short) != 0:
            out[i] = 0
        else:
            for j in range(len_short, len_long):
                if in_long[j] != b'\0':
                    out[i] = 0
                    break
            else:
                # all characters are '\0', so the strings match.
                out[i] = 1
            out[i] = 0

        in_long += stride_long
        in_short += stride_short


# Statically declare spec (can be discarted later):
cdef dtype_api.PyArrayMethod_Spec spec
spec.name = "string_equal"
spec.nin = 2
spec.nout = 1
cdef PyObject *dtypes[3]
spec.dtypes = dtypes
# Not used right now, but we can indicate not to check float errors:
spec.flags = dtype_api.NPY_METH_NO_FLOATINGPOINT_ERRORS
    

_string_equality_arraymethod = dtype_api.PyArrayMethod_FromSpec(&spec)



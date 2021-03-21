#cython: language_level=3

from . cimport dtype_api


dtype_api.import_experimental_dtype_api(0)



# cdef int multiple_uint8() with nogil:

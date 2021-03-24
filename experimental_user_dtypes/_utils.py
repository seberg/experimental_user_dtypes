import numpy as np
from textwrap import dedent as _dedent


def make_binary_ufunclike(method, name=None, module=None):
    """
    Wraps an ArrayMethod to behave almost exactly like a binary ufunc that
    only works for the specific DTypes.
    Note that this uses private (Python side) API currently.
    
    This function is provided, because it is currently not possible to
    register an ArrayMethod as a ufunc "inner-loop"
    """
    def func(arr1, arr2):
        casting, dtypes = method._resolve_descriptors(
                (arr1.dtype, arr2.dtype, None))
        # Could check casting, but it is not interesting usually.

        it = np.nditer((arr1, arr2, None), op_dtypes=dtypes,
                flags=["zerosize_ok", "grow_inner", "buffered", "external_loop"],
                op_flags=[["readonly"], ["readonly"], ["writeonly", "allocate"]])
        res = it.operands[2]
        with it:
            for op1, op2, out in it:
                method._simple_strided_call((op1, op2, out))
        
        return res

    if name:
        func.__name__ = name
    if module:
        func.__module__ = module
    func.__doc__ = _dedent(f"""
        Automatically generated ufunc-like for The ArrayMethod:
        {repr(method)}
        """)

    return func


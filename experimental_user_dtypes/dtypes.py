import numpy as np


# These are things we have to make nicely available in NumPy eventually
# (actually should probably use the C names, since that is technically more
# correct)
Int8 = type(np.dtype("int8"))
Uint8 = type(np.dtype("uint8"))
Int16 = type(np.dtype("int16"))
Uint16 = type(np.dtype("uint16"))
Int32 = type(np.dtype("int32"))
Uint32 = type(np.dtype("uint32"))
Int64 = type(np.dtype("int64"))
Uint64 = type(np.dtype("uint64"))

Float16 = type(np.dtype("e"))
Float32 = type(np.dtype("f"))
Float64 = type(np.dtype("d"))


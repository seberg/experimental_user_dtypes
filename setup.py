import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np
print(np.__version__)


ext_modules=[
    Extension("experimental_user_dtypes.scaled_uint8",
            ["experimental_user_dtypes/scaled_uint8.pyx"],
            include_dirs=[np.get_include()],
            ),
    Extension("experimental_user_dtypes.string_funcs",
            ["experimental_user_dtypes/string_funcs.pyx"],
            include_dirs=[np.get_include()],
            ),
    ]

setup(
  name="experimental_user_dtypes",
  cmdclass={"build_ext": build_ext},
  ext_modules=ext_modules)

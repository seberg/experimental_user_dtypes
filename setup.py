import os
import setuptools
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np
print(np.__version__)


ext_modules=[
    Extension("experimental_user_dtypes.float64unit",
            ["experimental_user_dtypes/float64unit.pyx"],
            include_dirs=[np.get_include()],
            ),
    Extension("experimental_user_dtypes.string_funcs",
            ["experimental_user_dtypes/string_funcs.pyx"],
            include_dirs=[np.get_include()],
            ),
    Extension(
        "experimental_user_dtypes.rational",
        ["experimental_user_dtypes/rational.c"],
        include_dirs=[np.get_include()],
    ),
    ]


if __name__ == "__main__":
    setup(
      name="experimental_user_dtypes",
      cmdclass={"build_ext": build_ext},
      packages=setuptools.find_packages(),
      ext_modules=ext_modules)


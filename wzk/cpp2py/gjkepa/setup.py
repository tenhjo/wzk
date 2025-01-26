import os
import platform
from setuptools import Extension, setup

file = os.path.abspath(os.path.dirname(__file__))
print(file)

if platform.system() == 'Linux':
    CPP_BULLET_INC = '/home/tenh_jo/USERDIR/Software/vcpkg/packages/bullet3_x64-linux/include/bullet/'
    CPP_BULLET_LIB = '/home/tenh_jo/USERDIR/Software/vcpkg/packages/bullet3_x64-linux/lib/'
else:
    CPP_BULLET_INC = '/usr/local/Cellar/bullet/3.08_2/include/bullet/'
    CPP_BULLET_LIB = '/usr/local/Cellar/bullet/3.08_2/lib/'


ext = Extension(
    name='gjkepa',
    sources=['./gjkepa.cpp'],
    extra_compile_args=['-std=c++1y', '-ffast-math', '-Ofast', '-fpermissive'],
    include_dirs=[file],
    library_dirs=[],
    libraries=[],
    language='c++',
)

setup(
    name='gjkepa',
    version='0.1.0',
    ext_modules=[ext],
)

# python setup.py develop


# TODO install bullet and see if it works
#   Alternative https://github.com/andreacasalino/Flexible-GJK-and-EPA

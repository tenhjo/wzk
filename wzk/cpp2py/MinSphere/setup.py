import os
from setuptools import Extension, setup


HOMEBREW_PREFIX = "/opt/homebrew/Cellar"
BOOST_DIR = f"{HOMEBREW_PREFIX}/boost"
CGAL_DIR = f"{HOMEBREW_PREFIX}/cgal"


CONDA_PREFIX = os.environ.get("CONDA_PREFIX")

boost_include_dir = f"{CONDA_PREFIX}/include/boost"
cgal_include_dir = f"{CONDA_PREFIX}/include/CGAL22"  # why does this work??

# TODO add a switch here
# boost_include_dir = os.environ.get("BOOST_INCLUDE_DIR",  # either user-defined
#                                   f"{BOOST_DIR}/{files.listdir(BOOST_DIR)[0]}/include")  # or installed via homebrew
# cgal_include_dir = os.environ.get("CGAL_INCLUDE_DIR",  # either user-defined
#                                  f"{CGAL_DIR}/{files.listdir(CGAL_DIR)[0]}/include")  # or installed via homebrew

ext = Extension(
    name="wzkMinSphere",
    sources=["./MinSphere.cpp"],
    extra_compile_args=["-std=c++1y", "-ffast-math", "-Ofast", "-fpermissive"],
    include_dirs=[boost_include_dir, cgal_include_dir],
    library_dirs=[],
    libraries=[],
    language="c++",
)
#
setup(
    name="wzkMinSphere",
    version="0.1.0",
    ext_modules=[ext],
)

# # '/opt/homebrew/Cellar/cgal/5.5.1/include'
# # '/opt/homebrew/Cellar/boost/1.81.0/include'
# pip install -e .  # TODO add to auto installer for wzk

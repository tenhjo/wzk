import os
import sys

from subprocess import call
from importlib import metadata


def upgrade_all():
    packages = [dist.name for dist in metadata.distributions()]
    call("pip install --upgrade " + " ".join(packages), shell=True)


def get_python_version():
    return sys.version_info[0], sys.version_info[1]


def get():
    pass


def get_conda_site_packages():
    python_xy = f"python{sys.version_info[0]}{sys.version_info[1]}"
    cs = os.environ.get("CONDA_PREFIX") + f"/lib/{python_xy}/site-packages"
    return cs


def get_conda_include():
    ci = os.environ.get("CONDA_PREFIX") + "/include"
    return ci

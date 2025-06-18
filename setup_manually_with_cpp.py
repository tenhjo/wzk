import os
import subprocess

verbose = 1

# Setup installation directories
directory0 = os.path.abspath(os.path.dirname(__file__))
if directory0 == "":
    directory0 = "."
directory0 = f"{directory0}/wzk/cpp2py"

directory_list = [f"{directory0}/MinSphere",  # TODO move gjkepa and minsphere to mopla
                  f"{directory0}/gjkepa"]


for directory in directory_list:
    quiet = f" > {os.devnull}" if verbose >= 1 else ""
    subprocess.call(f"cd {directory}; pip install -e . {quiet}", shell=True)

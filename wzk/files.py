import os

import re
import pickle
import json
import shutil
from typing import Union
import platform
import subprocess
import msgpack

import numpy as np
from scipy.io import loadmat as load_mat, savemat as save_mat  # noqa: F401

from wzk import time2, printing, subprocess2


__open_cmd_dict = {"Linux": "xdg-open",
                   "Darwin": "open",
                   "Windows": "start"}

# ICLOUD = 'Library/Mobile Documents/com~apple~CloudDocs'

EXT_DICT = dict(pkl="pkl", pickle="pickle",
                json="json",
                txt="text", text="txt",
                mat="mat",
                msgpack="msgpack")


def get_pythonpath():
    try:
        return os.environ["PYTHONPATH"].split(os.pathsep)
    except KeyError:
        return []


# --- shell ------------------------------------------------------------------------------------------------------------
def cp(src, dst, a=False):
    """-a (improved recursive copy, including all files, sub-folders and symlinks)"""
    if a:
        if not src.endswith("/."):
            src += "/."
        subprocess.call(f"cp -a {src} {dst}", shell=True)
    else:
        subprocess.call(f"cp {src} {dst}", shell=True)


def mv(src, dst):
    if src == dst:
        print(f"mv: src == dst | {src}")
        return

    subprocess.call(f"mv {src} {dst}", shell=True)


def rm(file):
    try:
        os.remove(file)
    except FileNotFoundError:
        pass


def rmdirs(directory: Union[str, list]):
    if isinstance(directory, list):
        for d in directory:
            rmdirs(d)
    else:
        if os.path.exists(directory):
            shutil.rmtree(path=directory)
        else:
            pass


def rm_files_in_dir(directory: str, file_list: list = None):
    if file_list is None:
        if os.path.exists(directory):
            file_list = listdir(directory)
        else:
            return

    for file in file_list:
        rm(os.path.join(directory, file))


def rm_empty_folders(directory):
    for d, _, _ in os.walk(directory, topdown=False):
        if len(os.listdir(d)) == 0:
            print(d)
            os.rmdir(d)


def rm_files_for_each(directory, file_list):
    directory_list = helper__get_sub_directory_list(directory=directory)
    for d in directory_list:
        rm_files_in_dir(directory=d, file_list=file_list)


def mkdirs(directory: Union[str, list]):
    if directory is None or directory == "":
        return

    if isinstance(directory, list):
        for d in directory:
            mkdirs(d)
    else:
        os.makedirs(directory, exist_ok=True)


def mkdir_for_each(directory, new_sub_directory):
    new_sub_directory = os.path.normpath(new_sub_directory)
    directory_list = helper__get_sub_directory_list(directory=directory)

    directory_list = [f"{os.path.normpath(d)}/{new_sub_directory}" for d in directory_list]
    mkdirs(directory_list)


def replace_with_link(src, file_list):
    """
    remove all files in the list and link them to src
    """
    for file in file_list:
        rm(file)
        os.symlink(src=src, dst=file)


def __read_head_tail(file: str,
                     n: int = 1,
                     squeeze: bool = True,
                     head_or_tail: str = "head"):
    assert head_or_tail == "head" or head_or_tail == "tail"
    s = os.popen(f"{head_or_tail} -n {n} {file}").read()
    s = s.split("\n")[:-1]

    if squeeze and len(s) == 1:
        s = s[0]

    return s


def read_head(file: str, n: int = 1, squeeze: bool = True):
    return __read_head_tail(file=file, n=n, squeeze=squeeze, head_or_tail="head")


def read_tail(file: str, n: int = 1, squeeze: bool = True):
    return __read_head_tail(file=file, n=n, squeeze=squeeze, head_or_tail="tail")


def listdir(directory: str, only_files=False, sort=True,
            ignore_files=(".DS_Store",), ignore_patterns=None):

    fl = os.listdir(directory)
    if only_files:
        fl = [f for f in fl if os.path.isfile(os.path.join(directory, f))]

    fl = [f for f in fl if f not in ignore_files]

    if ignore_patterns is not None:
        for p in ignore_patterns:
            fl = [f for f in fl if not re.search(p, f)]

    if sort:
        fl = sorted(fl)

    return fl


def list_directories(directory: str):
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]


def ensure_extension_point(ext: str):
    if ext[0] != ".":
        ext = "." + ext
    return ext


def ensure_file_extension(file: str, ext: str):
    ext = ensure_extension_point(ext)

    if file[-len(ext)] != ext:
        idx_dot = file[::-1].find(".")
        if idx_dot != -1:
            file = file[:-(idx_dot+1)]
        file += ext

    return file


def remove_extension(file: str, ext: str):
    ext = ensure_extension_point(ext)
    file = ensure_file_extension(file=file, ext=ext)
    file = file[:-len(ext)]
    return file


# –-- pickle -----------------------------------------------------------------------------------------------------------
def save_pickle(obj, file: str):
    file = ensure_file_extension(file=file, ext=EXT_DICT["pkl"])
    mkdirs(directory=os.path.split(file)[0])
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(file: str):
    ext_list = [EXT_DICT["pkl"], EXT_DICT["pickle"]]

    for ext in ext_list:
        file = ensure_file_extension(file=file, ext=ext)
        if os.path.exists(file):
            with open(file, "rb") as f:
                obj = pickle.load(f)
            return obj


# json
def save_json(obj, file: str):
    file = ensure_file_extension(file=file, ext=EXT_DICT["json"])
    mkdirs(directory=os.path.split(file)[0])
    with open(file, "w") as f:
        json.dump(obj, f, indent=4, sort_keys=True)


def load_json(file: str):
    file = ensure_file_extension(file=file, ext=EXT_DICT["json"])
    with open(file, "r") as f:
        obj = json.load(f)
    return obj


# msgpack
def load_msgpack(file):
    with open(file, "rb") as f:
        b = f.read()
    return msgpack.unpackb(b)


def save_msgpack(file, nested_list):
    arr_bin = msgpack.packb(nested_list, use_bin_type=True)
    mkdirs(directory=os.path.split(file)[0])
    with open(file, "wb") as f:
        f.write(arr_bin)


# txt
def save_object2txt(obj, file: str):
    file = ensure_file_extension(file=file, ext="txt")
    mkdirs(directory=os.path.split(file)[0])
    with open(file, "w") as f:
        f.write("".join(["%s: %s\n" % (k, v) for k, v in obj.__dict__.items()]))


# --- NPY --------------------------------------------------------------------------------------------------------------
# *.npy and *.npz files (maybe own module)
def combine_npz_files(*, directory,
                      pattern=None, file_list=None,
                      save=True,
                      verbose=0):

    if file_list is None:
        if pattern is None:
            pattern = re.compile(pattern=r"^[\S]+.npz$")

        assert isinstance(pattern, str)
        assert ".npz" in pattern
        pattern = re.compile(pattern=pattern)

        file_list = listdir(directory=directory, only_files=True)
        file_list = sorted([file for file in file_list if pattern.match(file)])

    new_dict = {}

    for i, file in enumerate(file_list):
        if verbose > 0:
            printing.progress_bar(i=i, n=len(file_list))

        data = np.load(directory + file)
        if i == 0:
            for key in data:
                new_dict[key] = data[key]
        else:
            for key in data:
                new_dict[key] = np.concatenate([new_dict[key], data[key]])

    if save:
        np.savez(directory + "combined_" + time2.get_timestamp() + ".npz", **new_dict)
    return new_dict


def combine_npy_files2(directory: str,
                       new_name: str = "combined_{new_len}") -> None:
    directory = os.path.normpath(path=directory)
    file_list = [file for file in os.listdir(directory) if ".npy" in file]
    arr = np.concatenate([np.load(f"{directory}/{file}", allow_pickle=False)
                          for file in file_list], axis=0)
    np.save(file=f"{directory}/{new_name.format(new_len=len(arr))}.npy", arr=arr)


def combine_npy_files(directory: str,
                      new_name: str = "combined_{new_len}",
                      delete_singles: bool = False,
                      verbose: int = 0) -> np.ndarray:

    directory = os.path.normpath(path=directory)
    file_list = [file for file in os.listdir(directory) if ".npy" in file]
    if verbose:
        print(file_list)

    arr = np.concatenate([np.load(f"{directory}/{file}", allow_pickle=True)[np.newaxis, :]
                          for file in file_list], axis=0)

    if arr.dtype.hasobject:
        arr = [np.concatenate(a) if isinstance(a[0], (tuple, list, np.ndarray)) else a for a in arr.T]
        np.save(file=f"{directory}/{new_name.format(new_len=len(arr[0]))}.npy", arr=arr)
    else:
        arr = arr.reshape([-1] + list(arr.shape)[2:])
        np.save(file=f"{directory}/{new_name.format(new_len=len(arr))}.npy", arr=arr)

    if delete_singles:
        for file in file_list:
            rm(file)
    return arr


def clip_npz_file(n_samples: int,
                  file: str,
                  save: bool = True) -> dict:
    directory, file = os.path.split(file)
    file_name, file_extension = os.path.splitext(file)
    assert file_extension == ".npz"

    data = np.load(directory + file)
    new_dict = {}
    for key in data:
        new_dict[key] = data[key][:n_samples]
    if save:
        np.savez(directory + file_name + "clipped_" + time2.get_timestamp() + file_extension, **new_dict)
    return new_dict


# --- start ---
def start_open(file: str):
    open_cmd = __open_cmd_dict[platform.system()]
    subprocess.Popen([f"{open_cmd} {file}"], shell=True)


def copy2clipboard(file: str):
    """
    https://apple.stackexchange.com/questions/15318/using-terminal-to-copy-a-file-to-clipboard
    -> works only for mac!
    """
    subprocess.run(["osascript",
                    "-e",
                    'set the clipboard to POSIX file "{}"'.format(file)])


# --- chmod ---
def chmod_file(file, mod):
    subprocess2.call2(cmd=f"sudo chmod {mod} {file}")


def chmod_dir(directory, mod):
    subprocess2.call2(cmd=f"sudo chmod {mod} -R {directory}")


def are_files_identical(file_a, file_b):
    with open(file_a, "rb") as f:
        aa = f.read()

    with open(file_b, "rb") as f:
        bb = f.read()

    return aa == bb


# --- Directory Magic --------------------------------------------------------------------------------------------------
def split_files_into_dirs(file_list: list,
                          bool_fun,
                          dir_list: list,
                          base_dir: str = None,
                          mode: str = "dry"):

    if base_dir is not None:
        base_dir = os.path.normpath(base_dir)
    else:
        base_dir = ""

    if file_list is None and base_dir:
        file_list = os.listdir(base_dir)
        print(f"Get file_list from {base_dir}")

    for i, d_i in enumerate(dir_list):
        d_i = os.path.normpath(d_i)

        print(f"->{d_i}")

        j = 0
        while j < len(file_list):
            f_j = file_list[j]

            if bool_fun(f_j, i):
                f_j = os.path.normpath(f_j)
                f_j_new = f"{d_i}/{os.path.split(f_j)[-1]}"

                if mode == "wet":
                    shutil.move(f"{base_dir}/{f_j}", f_j_new)
                print(f_j)

                file_list.pop(j)
            else:
                j += 1

    if mode != "wet":
        print()
        print("'dry' mode is activated by default, to apply the changes use mode='wet')")


def dir_dir2file_array(directory: str = None,
                       combine_str: bool = True) -> list:
    """
    -directory/
    ----subA/
    --------fileA1
    --------fileA2
    --------fileA3
    ----subB/
    --------fileB1
    --------fileB2

    # combined_str = False
    -> [[fileA1, fileA2, fileA3],
        [fileB1, fileB2]]
    # combined_str = True
    -> [[directory/subA/fileA1, directory/subA/fileA2, directory/subA/fileA3],
        [directory/subB/fileB1, directory/subB/fileB2]]
    """

    if directory is None:
        directory = os.getcwd()
    dir_list = sorted([d for d in os.listdir(directory) if d[0] != "."])

    file_arr = []
    for dir_i in dir_list:
        f_list = sorted([f for f in os.listdir(directory + "/" + dir_i) if f[0] != "."])

        if combine_str:
            f_list = [f"{directory}/{dir_i}/{f}" for f in f_list]

        file_arr.append(f_list)

    return file_arr


def rename_directories_inbetween(directory, inbetweens, new_inbetweens=""):
    for directory_i, directory_list, file_list in os.walk(directory):
        for f in file_list:
            old = f"{directory_i}/{f}"
            new = old.replace(inbetweens, new_inbetweens)
            mv(old, new)


def get_sub_directories(directory):
    directory = os.path.normpath(directory)
    subs = next(os.walk(directory))[1]
    subs = [f"{directory}/{s}" for s in subs]
    return subs


def helper__get_sub_directory_list(directory):
    if isinstance(directory, str):
        directory_list = get_sub_directories(directory=directory)
    else:
        directory_list = os.listdir(directory)
    return directory_list

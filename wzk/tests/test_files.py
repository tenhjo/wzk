from wzk import files


def test_split_files_into_dirs():

    n, m = 20, 3
    from wzk.strings import uuid4

    file_list = [f"{i}{uuid4()}" for i in range(n) for _ in range(m)]
    dirs = [f"new_{i}" for i in range(n)]

    files.split_files_into_dirs(
        file_list=file_list,
        dir_list=dirs,
        mode="dry",
        bool_fun=lambda s, i: (s[: len(str(i))] == str(i)) and len(s) == 32 + len(str(i)),
    )

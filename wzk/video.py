import os
import numpy as np

import fire

from wzk import files, strings


def stack_videos(videos=None, file=None):
    """
    https://stackoverflow.com/questions/11552565/vertically-or-horizontally-stack-mosaic-several-videos-using-ffmpeg/33764934#33764934
    """
    if isinstance(videos, str):
        videos = files.dir_dir2file_array(videos)

    _format = "mp4"
    kwargs = ""

    s0, s1 = np.shape(videos)

    if file is None:
        file = f"stacked_video__{s0}x{s1}__{strings.uuid4()}.{_format}"

    uuid_list = [f"{strings.uuid4()}.{_format}" for _ in range(s0)]

    for i, in_i in enumerate(videos):
        in_i_str = " -i ".join(in_i)
        stack_str = "".join([f"[{j}:v]" for j in range(s1)])

        os.system(f"ffmpeg -i {in_i_str} "
                  f"{kwargs} "
                  f'-filter_complex "{stack_str}"hstack=inputs={s1}[v] -map "[v]" '
                  f"{uuid_list[i]}")

    in_i_str = " -i ".join(uuid_list)
    stack_str = "".join([f"[{j}:v]" for j in range(s0)])
    os.system(f"ffmpeg -i {in_i_str} "
              f"{kwargs} "
              f'-filter_complex "{stack_str}"vstack=inputs={s0}[v] -map "[v]" '
              f"{file}")

    for u in uuid_list:
        os.remove(u)


if __name__ == "__main__":
    fire.Fire(stack_videos)

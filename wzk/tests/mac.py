import os
import numpy as np


def send_key(key_code):
    assert isinstance(key_code, int)
    os.system(f'osascript -e \'tell application "System Events"\' -e \'key code {key_code}\' -e \' end tell\'')


def change_brightness(brightness):
    assert isinstance(brightness, int)
    assert -16 <= brightness <= 16

    if brightness > 0:
        key_code = 144
    else:
        key_code = 145

    for _ in range(np.abs(brightness)):
        send_key(key_code=key_code)


def visual_bell(n=3):
    change_brightness(-n)
    change_brightness(+n)


if __name__ == "__main__":
    visual_bell(n=10)

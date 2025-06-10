import numpy as np
from scipy.spatial.transform import Rotation as R
from wzk import spatial


def sample_24cube_rotations(n):
    # 6 x 4
    rots = R.create_group("O")
    dcm24 = rots.as_matrix()
    e24 = np.rad2deg(spatial.dcm2euler(dcm=dcm24, seq="zxz"))
    i = np.random.randint(low=0, high=24, size=n)
    return e24[i]
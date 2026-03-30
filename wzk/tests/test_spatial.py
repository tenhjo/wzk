
from wzk.logger import log_print
from unittest import TestCase

import numpy as np
from wzk import spatial


class Test(TestCase):
    def test_a(self):
        from wzk.spatial.difference import frame_difference
        a = spatial.sample_frames(x_low=np.zeros(3), x_high=np.ones(3))
        b = spatial.sample_frames(x_low=np.zeros(3), x_high=np.ones(3))

        a[:3, :3] = spatial.sample_dcm_noise(shape=1, scale=1e-8)
        b[:3, :3] = spatial.sample_dcm_noise(shape=1, scale=1e-8)

        c = spatial.invert(a) @ b
        c_rc = spatial.frame2rotvec(c)
        log_print("rad c ", np.linalg.norm(c_rc))

        ab_trans, ab_rot = frame_difference(a, b)
        log_print("rad ab", ab_rot)

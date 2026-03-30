from unittest import TestCase

import numpy as np

from wzk import image


class Test(TestCase):
    def test_img2compressed(self):
        size = (64, 35)
        n_dim = 2
        img = np.random.random(size)
        img_cmp = image.img2compressed(
            img=img,
            n_dim=n_dim,
        )
        img2 = image.compressed2img(img_cmp=img_cmp, shape=size, n_dim=n_dim, dtype=float)

        self.assertTrue(np.allclose(img, img2))

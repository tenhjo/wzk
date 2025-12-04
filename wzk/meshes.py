import numpy as np


def quad_to_tri_faces(faces4):
    faces4 = np.asarray(faces4)
    v0 = faces4[:, 0]
    v1 = faces4[:, 1]
    v2 = faces4[:, 2]
    v3 = faces4[:, 3]

    faces3_a = np.stack([v0, v1, v2], axis=-1)
    faces3_b = np.stack([v0, v2, v3], axis=-1)
    faces3 = np.concatenate([faces3_a, faces3_b], axis=0)
    return faces3

import numpy as np
import trimesh

from wzk import grid


def bimg2trimesh(img, limits, colors=(0.8, 0.8, 0.8, 1.0)):
    voxel_size = grid.limits2voxel_size(shape=img.shape, limits=limits, unify=False)
    transform = np.eye(4)
    transform[:3, 3] = limits[:, 0] + voxel_size / 2
    transform[range(3), range(3)] = voxel_size
    mesh = trimesh.voxel.VoxelGrid(img, transform=transform)
    mesh = mesh.as_boxes(colors=colors)
    return mesh

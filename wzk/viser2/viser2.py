import numpy as np
import trimesh

import viser
from viser import SceneApi, LineSegmentsHandle

from wzk import grid

from typing import (Tuple, Union)
from typing_extensions import TypeAlias


RgbTupleOrArray: TypeAlias = Union[
    Tuple[int, int, int], Tuple[float, float, float], np.ndarray
]


def bimg2trimesh(img, limits, colors=(0.8, 0.8, 0.8, 1.0)):
    voxel_size = grid.limits2voxel_size(shape=img.shape, limits=limits, unify=False)
    transform = np.eye(4)
    transform[:3, 3] = limits[:, 0] + voxel_size / 2
    transform[range(3), range(3)] = voxel_size
    mesh = trimesh.voxel.VoxelGrid(img, transform=transform)
    mesh = mesh.as_boxes(colors=colors)
    return mesh

def points_toN23(points: np.ndarray,
                 flatten: bool = True) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if (
            points.shape[-1] != 3
            or points.ndim != 3
            or points.shape[1] != 2
    ):
        shape2 = np.array(points.shape[:-1] + (2, 3))
        shape2[-3] -= 1
        points2 = np.zeros(shape2)
        points2[..., 0, :] = points[..., :-1, :]
        points2[..., 1, :] = points[..., 1:, :]
    else:
        points2 = points

    if flatten:
        points2 = points2.reshape(-1, 2, 3)

    return points2


def add_line_segments2(scene: SceneApi,
                       name: str,
                       points: np.ndarray,
                       colors: np.ndarray | RgbTupleOrArray) -> LineSegmentsHandle:

    points = points_toN23(points=points, flatten=True)
    return scene.add_line_segments(name=name, points=points, colors=colors)


def fov(h_deg, v_deg, z_min_m, z_max_m, num_slices=2):
    # TODO move to better module
    tan_h = np.tan(np.deg2rad(h_deg / 2.0))  # ≈ 0.949
    tan_v = np.tan(np.deg2rad(v_deg / 2.0))

    def rect_at_z(z):
        x = tan_h * z
        y = tan_v * z
        corners = np.array([[-x,  y, z],
                            [ x,  y, z],
                            [ x, -y, z],
                            [-x, -y, z]])
        return corners

    # --- Generate slices along z ---
    z_list = np.linspace(z_min_m, z_max_m, num_slices)

    # Draw rays from origin to far-plane corners (frustum edges)
    edges = np.zeros((4, 2, 3))
    edges[:, 1, :] = rect_at_z(z_max_m)

    planes = []
    for z_i in z_list:
        rect = np.zeros((4, 2, 3))
        rect[:, 0, :] = rect_at_z(z=z_i)
        rect[:, 1, :] = np.roll(rect[:, 0, :], axis=0, shift=1)
        planes.append(rect)
    planes = np.concatenate(planes, axis=0)
    edges_planes = np.concatenate([edges, planes], axis=0)
    return edges_planes

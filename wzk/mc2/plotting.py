import os

import numpy as np

import meshcat
from meshcat import geometry as mg, transformations as mt

from wzk import grid, bimage, mpl2, spatial, geometry, np2, uuid4

Visualizer = meshcat.Visualizer
MeshGeometry_DICT = dict(stl=mg.StlMeshGeometry,
                         obj=mg.ObjMeshGeometry,
                         dae=mg.DaeMeshGeometry)

default_color = "white"


def rgba2material(rgba, material=None):
    # TODO can I set linewidth etc. also here?
    if material is None:
        material = mg.MeshPhongMaterial()

    if rgba is not None:
        material.color = mpl2.colors2.rgb2hex(rgb=rgba[:3])
        if float(rgba[3]) != 1.0:
            material.transparent = True
            material.opacity = float(rgba[3])

    else:
        material.color = None

    return material


def get_material(color=default_color, alpha=1.0, wireframe=False):
    rgba = mpl2.colors.to_rgba(c=color, alpha=alpha)

    # material = mg.MeshPhongMaterial()
    # material = mg.MeshBasicMaterial()
    material = mg.MeshLambertMaterial(dict(flatShading=True))
    material = rgba2material(rgba=rgba, material=material)
    material.wireframe = wireframe
    return material


def wrapper_handle(vis, handle=None, default="", n=None):
    if handle is None:
        if n is None:
            return f"{default}-{uuid4()}"
        else:
            return [wrapper_handle(vis=vis, handle=handle, default=default, n=None) for _ in range(n)]

    elif isinstance(handle, str):
        pass

    elif isinstance(handle, list):
        if len(handle) == n:
            pass

        elif len(handle) > n:
            delete(vis=vis, handle=handle[n:])
            handle = handle[:n]

        elif len(handle) < n:
            handle += wrapper_handle(vis=vis, handle=None, default=default, n=n-len(handle))

    return handle


def wrapper_x(x):
    assert x.ndim == 2 and x.shape[1] == 3
    return x.astype(np.float32).T


def wrapper_faces(faces):
    assert faces.ndim == 2
    if faces.shape[1] == 4:
        faces = geometry.faces4_to_3(f4=faces)
    assert faces.shape[1] == 3
    return faces.astype(int)


def color2rgb_list(color, n):
    rgb = mpl2.colors.to_rgba(c=color, alpha=1)
    rgb_list = np.repeat(np.array(rgb)[np.newaxis, :], repeats=n, axis=0)
    return rgb_list.astype(np.float32).T


def plot_points(vis, h,
                x, size=0.001, color=default_color):

    h = wrapper_handle(vis=vis, handle=h, default="points")
    x = wrapper_x(x)

    material = mg.PointsMaterial(size=size)
    rgb_list = color2rgb_list(color=color, n=x.shape[1])

    vis[h].set_object(geometry=mg.PointsGeometry(position=x, color=rgb_list), material=material)

    return h


def plot_lines(vis, h, x, lines=None, color=default_color, alpha=1.):
    h = wrapper_handle(vis=vis, handle=h, default="lines")
    x = wrapper_x(x)

    material = get_material(color=color, alpha=alpha)
    # material.linewidth = linewidth
    # material.wireframe = True
    # material.wireframeLinewidth = linewidth

    if lines is None:
        lines = mg.Line(geometry=mg.PointsGeometry(position=x, color=None), material=material)

    else:
        xl = x[:, lines].reshape(3, -1)
        lines = mg.LineSegments(geometry=mg.PointsGeometry(position=xl, color=None), material=material)

    vis[h].set_object(lines)

    return h


def plot_faces(vis, h, x, faces, color=default_color, alpha=1.0):
    h = wrapper_handle(vis=vis, handle=h, default="faces")
    faces = wrapper_faces(faces=faces)

    material = get_material(color=color, alpha=alpha)

    vis[h].set_object(geometry=mg.TriangularMeshGeometry(vertices=x, faces=faces), material=material)


def plot_spheres(vis, h, x, r, color=default_color, alpha=1.0, wireframe=False, **kwargs):  # noqa
    material = get_material(color=color, alpha=alpha, wireframe=wireframe)
    x = np.atleast_2d(x)
    r = np.atleast_1d(r)

    h = wrapper_handle(vis=vis, handle=h, default="spheres", n=len(x))
    assert len(x) == len(r) == len(h)

    for hh, xx, rr in zip(h, x, r):
        vis[hh].set_object(geometry=mg.Sphere(radius=rr), material=material)
        vis[hh].set_transform(mt.translation_matrix(xx))

    return h


def plot_cube(vis, h, limits, mode="faces", **kwargs):
    if limits is None:
        return
    v, e, f = geometry.cube(limits=limits)

    if mode == "faces":
        return plot_faces(vis=vis, h=h, x=v, faces=f, **kwargs)
    elif mode == "lines":
        return plot_lines(vis=vis, h=h, x=v, lines=e, **kwargs)
    else:
        raise ValueError


def plot_bimg_voxel(vis, h,
                    bimg, limits, color=default_color, alpha=1.0):
    h = wrapper_handle(vis=vis, handle=h, default="bimg")

    material = get_material(color=color, alpha=alpha)
    voxel_size = grid.limits2voxel_size(shape=bimg.shape, limits=limits)

    i = np.array(np.nonzero(bimg)).T
    x = grid.i2x(i=i, limits=limits, shape=bimg.shape, mode="c")
    for j, xx in enumerate(x):
        vis[f"{h}/voxel-{j}"].set_object(geometry=mg.Box([voxel_size] * 3), material=material)
        vis[f"{h}/voxel-{j}"].set_transform(mt.translation_matrix(xx))

    return h


def delete(vis, handle):
    if isinstance(handle, str):
        vis[handle].delete()
    elif isinstance(handle, list):
        for h in handle:
            vis[h].delete()


def plot_bimg_mesh(vis, h, bimg, limits, level=0, color=default_color, alpha=1.0):
    h = wrapper_handle(vis=vis, handle=h, default="bimg")

    material = get_material(color=color, alpha=alpha)

    voxel_size = grid.limits2voxel_size(shape=bimg.shape, limits=limits)
    v, f = bimage.bimg2surf(img=bimg, limits=limits + voxel_size / 2, level=level)
    print(np.sum(bimg))

    delete(vis=vis, handle=h)

    vis[h].set_object(geometry=mg.TriangularMeshGeometry(vertices=v, faces=f), material=material)
    return h


def plot_bimg(vis, h,
              img, limits, mode="mesh", **kwargs):
    if img is None:
        return

    if mode == "mesh":
        plot_bimg_mesh(vis=vis, h=h, bimg=img, limits=limits, **kwargs)

    elif mode == "voxel":
        plot_bimg_voxel(vis=vis, h=h, bimg=img, limits=limits, **kwargs)

    else:
        raise ValueError(f"Unknown mode: '{mode}' | ['mesh', 'voxel']")


def get_default_color_alpha(**kwargs):
    kwargs.update(dict(color=kwargs.pop("color", default_color)))
    kwargs.update(dict(alpha=kwargs.pop("alpha", 1.0)))
    return kwargs


def plot_arrow(vis, h, x, v, length=1.0, color=default_color, alpha=1.0):
    x, v = np2.squeeze_all(x, v)

    if np.ndim(x) == 2 or np.ndim(v) == 2:
        n = np2.max_size(x, v) // 3
        h, length, color, alpha = np2.scalar2array(h, length, color, alpha, shape=n)
        x, v = np2.scalar2array(x, v, shape=(n, 3))
        return [plot_arrow(vis=vis, h=hh, x=xx, v=vv, length=ll, color=cc, alpha=aa)
                for (hh, xx, vv, ll, cc, aa) in zip(h, x, v, length, color, alpha)]

    h = wrapper_handle(vis=vis, handle=h, default="arrow")
    h_cone = f"{h}-cone"
    h_cylinder = f"{h}-cylinder"

    scale_length2width = 0.05
    scale_length2cone_width = 0.1
    scale_length2cone_length = 0.3

    length_cone = length * scale_length2cone_length
    length_cylinder = length - length_cone

    radius_cylinder = length * scale_length2width
    radius_cone = length * scale_length2cone_width

    cylinder = mg.Cylinder(height=length_cylinder, radius=radius_cylinder)
    cone = mg.Cylinder(height=length_cone, radiusBottom=radius_cone, radiusTop=0)

    material = get_material(color=color, alpha=alpha)
    vis[h_cylinder].set_object(geometry=cylinder, material=material)
    vis[h_cone].set_object(geometry=cone, material=material)

    vy = v.astype(float)
    vx = geometry.get_orthonormal(vy)
    vz = np.ones(3)
    dcm = geometry.make_rhs(xyz=np.vstack([vx, vy, vz])).T

    f_cylinder = spatial.trans_dcm2frame(trans=x, dcm=dcm) @ spatial.trans2frame(y=length_cylinder / 2)
    f_cone = spatial.trans_dcm2frame(trans=x, dcm=dcm) @ spatial.trans2frame(y=length_cylinder + length_cone / 2)

    vis[h_cone].set_transform(f_cone)
    vis[h_cylinder].set_transform(f_cylinder)

    return h


def plot_coordinate_frames(vis, h, f, scale=1.0, **kwargs):
    xyz_str = "xyz"

    if np.ndim(f) == 2:
        h = wrapper_handle(vis=vis, handle=h, default="frame")

        color = kwargs.pop("color", ("red", "green", "blue"))
        color = np2.scalar2array(color, shape=3)

        for i in range(3):
            plot_arrow(vis=vis, h=f"{h}-{xyz_str[i]}", x=f[:3, -1], v=f[:3, i], length=scale, color=color[i], **kwargs)

        return h

    elif np.ndim(f) == 3:

        n = len(f)
        h = np2.scalar2array(h, shape=n)

        color = kwargs.pop("color", ("red", "green", "blue"))
        color = np2.scalar2array(color, shape=(n, 3))

        return [plot_coordinate_frames(vis=vis, h=hh, f=ff, color=cc, scale=scale, **kwargs)
                for hh, ff, cc in zip(h, f, color)]


def transform(vis, h, f):
    if f is None:
        return

    if isinstance(h, str) and f.ndim == 2:
        vis[h].set_transform(f)

    elif isinstance(h, (list, tuple, np.ndarray)) and f.ndim == 2:
        for hh in zip(h):
            vis[hh].set_transform(f)

    elif f.ndim == 3:
        assert len(h) == len(f)
        for hh, ff in zip(h, f):
            vis[hh].set_transform(ff)

    else:
        raise ValueError


def __load_mesh(mesh):
    ext = os.path.splitext(mesh)[-1][1:]
    return MeshGeometry_DICT[ext].from_file(mesh)


def plot_meshes(vis, h, meshes, f=None, color="white", alpha=1., **kwargs):  # noqa
    material = get_material(color=color, alpha=alpha)

    for hh, mm in zip(h, meshes):
        vis[hh].set_object(__load_mesh(mm), material)

    transform(vis=vis, h=h, f=f)
    return h

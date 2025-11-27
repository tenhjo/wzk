import os

import numpy as np

import meshcat
from meshcat import geometry as mg, transformations as mt

from wzk import grid, bimage, mpl2, spatial, geometry, np2, strings

Visualizer = meshcat.Visualizer
MeshGeometry_DICT = dict(stl=mg.StlMeshGeometry,
                         obj=mg.ObjMeshGeometry,
                         dae=mg.DaeMeshGeometry)

default_color = "white"

# TODO linewidth can apparently not be changed


def ih_visualizer(vis: Visualizer):
    if vis is None:
        vis = Visualizer(vis)
    return vis


def ih_handle(p: Visualizer, h=None, default="", n=None):
    if h is None:
        if n is None:
            return f"{default}-{strings.uuid4()}"
        else:
            return [ih_handle(p=p, h=h, default=default, n=None) for _ in range(n)]

    elif isinstance(h, str):
        pass

    elif isinstance(h, list):
        if len(h) == n:
            pass

        elif len(h) > n:
            delete(p=p, handle=h[n:])
            h = h[:n]

        elif len(h) < n:
            h += ih_handle(p=p, h=None, default=default, n=n - len(h))

    return h


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


def get_material(color=default_color, alpha: float = 1.0, wireframe: bool = False):
    rgba = mpl2.colors.to_rgba(c=color, alpha=alpha)

    # material = mg.MeshPhongMaterial()
    # material = mg.MeshBasicMaterial()
    material = mg.MeshLambertMaterial(dict(flatShading=True))
    material = rgba2material(rgba=rgba, material=material)
    material.wireframe = wireframe
    return material

def get_camera(vis):
    pass



def set_camera(vis: Visualizer, x, zoom: float = None):
    """camera is always oriented at the center"""

    # weird shift is necessary for meshcat
    x = np.array([x[0], +x[2], -x[1]])
    vis["/Cameras/default/rotated/<object>"].set_property("position", x.tolist())

    if zoom is not None:
        vis["/Cameras/default/rotated/<object>"].set_property("zoom", zoom)


def turn_grid_on_off(vis: Visualizer, on=False):
    vis["/Grid"].set_property("visible", on)


def turn_axes_on_off(vis: Visualizer, on=False):
    vis["/Axes"].set_property("visible", False)


def turn_background_on_off(vis: Visualizer, on=False):
    vis["/Background"].set_property("visible", False)


def turn_all_on_off(vis: Visualizer, on=False):
    turn_grid_on_off(vis=vis, on=on)
    turn_axes_on_off(vis=vis, on=on)
    turn_background_on_off(vis=vis, on=on)


def wrapper_x(x: np.ndarray):
    assert x.ndim == 2 and x.shape[1] == 3
    return x.astype(np.float32).T


def wrapper_faces(faces: np.ndarray):
    assert faces.ndim == 2
    if faces.shape[1] == 4:
        faces = geometry.faces4_to_3(f4=faces)
    assert faces.shape[1] == 3
    return faces.astype(int)


def color2rgb_list(color, n, alpha=1.0):
    rgb = mpl2.colors.to_rgba(c=color, alpha=alpha)
    rgb_list = np.repeat(np.array(rgb)[np.newaxis, :], repeats=n, axis=0)
    return rgb_list.astype(np.float32).T


def plot_points(x, size=0.001, color=default_color, alpha=1.0,
                vis: Visualizer = None, h=None):

    vis = ih_visualizer(vis)
    h = ih_handle(p=vis, h=h, default="points")
    x = wrapper_x(x)

    material = mg.PointsMaterial(size=size)
    rgb_list = color2rgb_list(color=color, n=x.shape[1], alpha=alpha)

    vis[h].set_object(geometry=mg.PointsGeometry(position=x, color=rgb_list), material=material)

    return h


def plot_lines(x, lines=None,
               color=default_color, alpha=1.,
               vis: Visualizer = None, h=None):

    vis = ih_visualizer(vis)
    h = ih_handle(p=vis, h=h, default="lines")

    x = wrapper_x(x)

    material = mg.LineBasicMaterial(color=color, vertexColors=False)

    if lines is None:
        ls = mg.Line(geometry=mg.PointsGeometry(position=x, color=None), material=material)

    else:
        x = x[:, lines].reshape(3, -1)
        ls = mg.LineSegments(geometry=mg.PointsGeometry(position=x, color=None), material=material)

    vis[h].set_object(ls)

    return h


def plot_faces(x, faces, color=default_color, alpha=1.0,
               vis: Visualizer = None, h=None):

    vis = ih_visualizer(vis)
    h = ih_handle(p=vis, h=h, default="faces")

    faces = wrapper_faces(faces=faces)

    material = get_material(color=color, alpha=alpha)

    vis[h].set_object(geometry=mg.TriangularMeshGeometry(vertices=x, faces=faces), material=material)


class SphereCustom(mg.Geometry):
    def __init__(self, radius: float, whSegments: int = 20):
        super(SphereCustom, self).__init__()
        self.radius: float = radius
        self.whSegments: int = whSegments

    def lower(self, object_data):
        return {
            u"uuid": self.uuid,
            u"type": u"SphereGeometry",
            u"radius": self.radius,
            u"widthSegments" : self.whSegments,
            u"heightSegments" : self.whSegments
        }


def plot_spheres(x, r, color=default_color, alpha=1.0, wireframe=False, whSegments=20,
                 vis: Visualizer = None, h=None,
                 **kwargs):  # noqa

    vis = ih_visualizer(vis)

    material = get_material(color=color, alpha=alpha, wireframe=wireframe)
    x = np.atleast_2d(x)
    r = np.atleast_1d(r)
    r = np2.scalar2array(r, shape=len(x))

    h = ih_handle(p=vis, h=h, default="spheres", n=len(x))
    assert len(x) == len(r) == len(h)

    for hh, xx, rr in zip(h, x, r):
        vis[hh].set_object(geometry=SphereCustom(radius=rr, whSegments=whSegments), material=material)
        vis[hh].set_transform(mt.translation_matrix(xx))

    return h


def plot_cube(limits, mode="faces",
              vis: Visualizer = None, h=None,
              **kwargs):
    if limits is None:
        return None

    v, e, f = geometry.cube(limits=limits)

    if mode == "faces":
        return plot_faces(vis=vis, h=h, x=v, faces=f, **kwargs)
    elif mode == "lines":
        return plot_lines(vis=vis, h=h, x=v, lines=e, **kwargs)
    else:
        raise ValueError


def plot_bimg_voxel(bimg, limits, color=default_color, alpha=1.0,
                    vis: Visualizer = None, h=None):

    vis = ih_visualizer(vis=vis)
    h = ih_handle(p=vis, h=h, default="bimg")

    material = get_material(color=color, alpha=alpha)
    voxel_size = grid.limits2voxel_size(shape=bimg.shape, limits=limits)

    i = np.array(np.nonzero(bimg)).T
    x = grid.i2x(i=i, limits=limits, shape=bimg.shape, mode="c")
    # plot_points(x=x, vis=vis, size=voxel_size/2)

    plot_spheres(x=x, vis=vis, r=np.ones(len(x)) * voxel_size / 2, alpha=alpha, color=color, whSegments=5)
    # for j, xx in enumerate(x):
    #     p[f"{h}/voxel-{j}"].set_object(geometry=mg.Box([voxel_size] * 3), material=material)
    #     p[f"{h}/voxel-{j}"].set_transform(mt.translation_matrix(xx))

    return h


def delete(p, handle):
    if isinstance(handle, str):
        p[handle].delete()
    elif isinstance(handle, list):
        for h in handle:
            p[h].delete()


def plot_bimg_mesh(bimg, limits,
                   level: float = 0, color=default_color, alpha=1.0,
                   vis: Visualizer = None, h=None):

    vis = ih_visualizer(vis=vis)
    h = ih_handle(p=vis, h=h, default="bimg")

    material = get_material(color=color, alpha=alpha)

    voxel_size = grid.limits2voxel_size(shape=bimg.shape, limits=limits)
    v, f = bimage.bimg2surf(img=bimg.astype(int), limits=limits + voxel_size / 2, level=level)

    delete(p=vis, handle=h)

    vis[h].set_object(geometry=mg.TriangularMeshGeometry(vertices=v, faces=f), material=material)
    return h


def plot_bimg(img, limits, mode="mesh",
              vis: Visualizer = None, h=None,
              **kwargs):

    vis = ih_visualizer(vis=vis)

    if img is None:
        return

    if mode == "mesh":
        plot_bimg_mesh(vis=vis, h=h, bimg=img, limits=limits, level=+0.55, **kwargs)

    elif mode == "voxel":
        plot_bimg_voxel(vis=vis, h=h, bimg=img, limits=limits, **kwargs)

    else:
        raise ValueError(f"Unknown mode: '{mode}' | ['mesh', 'voxel']")


def get_default_color_alpha(**kwargs):
    kwargs.update(dict(color=kwargs.pop("color", default_color)))
    kwargs.update(dict(alpha=kwargs.pop("alpha", 1.0)))
    return kwargs


def plot_arrow(x, v, length=1.0, color=default_color, alpha=1.0,
               vis: Visualizer = None, h=None):

    vis = ih_visualizer(vis=vis)

    x, v = np2.squeeze_all(x, v)

    if np.ndim(x) == 2 or np.ndim(v) == 2:
        n = np2.max_size(x, v) // 3
        h, length, color, alpha = np2.scalar2array(h, length, color, alpha, shape=n)
        x, v = np2.scalar2array(x, v, shape=(n, 3))
        return [plot_arrow(vis=vis, h=hh, x=xx, v=vv, length=ll, color=cc, alpha=aa)
                for (hh, xx, vv, ll, cc, aa) in zip(h, x, v, length, color, alpha)]

    h = ih_handle(p=vis, h=h, default="arrow")
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


def plot_coordinate_frames(f: np.ndarray,
                           p: Visualizer = None, h=None,
                           scale=1.0, color=("red", "green", "blue"), alpha=1.0):
    xyz_str = "xyz"

    p = ih_visualizer(vis=p)

    if np.ndim(f) == 2:
        h = ih_handle(p=p, h=h, default="frame")

        color, alpha = np2.scalar2array(color, alpha, shape=3)

        for i in range(3):
            plot_arrow(vis=p, h=f"{h}-{xyz_str[i]}", x=f[:3, -1], v=f[:3, i], length=scale,
                       color=color[i], alpha=alpha[i])

        return h

    elif np.ndim(f) == 3:

        n = len(f)
        h = np2.scalar2array(h, shape=n)

        color, alpha = np2.scalar2array(color, alpha, shape=(n, 3))

        return [plot_coordinate_frames(p=p, h=hh, f=ff, color=cc, alpha=aa, scale=scale)
                for hh, ff, cc, aa in zip(h, f, color, alpha)]


def transform(p: Visualizer, f: np.ndarray, h):
    if f is None:
        return
    f = np.array(f)

    if isinstance(h, str) and f.ndim == 2:
        p[h].set_transform(f)

    elif isinstance(h, (list, tuple, np.ndarray)) and f.ndim == 2:
        for h_i in h:
            p[h_i].set_transform(f)

    elif f.ndim == 3:
        assert len(h) == len(f)
        for h_i, f_i in zip(h, f):
            p[h_i].set_transform(f_i)

    else:
        raise ValueError


def load_mesh(file: str) -> mg.MeshGeometry:
    ext = os.path.splitext(file)[-1][1:]
    mesh = MeshGeometry_DICT[ext].from_file(file)
    return mesh


def plot_meshes(meshes, f: np.ndarray = None,
                vis: Visualizer = None,
                h=None,
                color="white", alpha=1., **kwargs):

    vis = ih_visualizer(vis=vis)
    material = get_material(color=color, alpha=alpha)

    if isinstance(meshes, (str, mg.MeshGeometry)):
        meshes = [meshes]

    h = ih_handle(p=vis, h=h, default="mesh", n=len(meshes))
    for h_i, m_i in zip(h, meshes):
        if not isinstance(m_i, mg.MeshGeometry):
            m_i = load_mesh(m_i)
        vis[h_i].set_object(m_i, material)

    transform(p=vis, h=h, f=f)
    return h


def save_png(vis, x_camera, file):
    if x_camera is not None:
        set_camera(vis=vis, x=x_camera)
    png = vis.get_image()
    png.save(file)

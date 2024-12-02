import numpy as np

from wzk import mc2, spatial


# ----------------------------------------------------------------------------------------------------------------------
def try_plot_points():
    n = 100000
    p = mc2.Visualizer()
    h = mc2.plot_points(p=p, h=None, x=np.random.random((n, 3)), color="white", size=0.01)
    input()
    mc2.plot_points(p=p, h=h, x=np.random.random((n, 3)), color="blue", size=0.02)


def try_plot_lines():
    n = 100
    p = mc2.Visualizer()
    mc2.plot_lines(p=p, h=None, x=np.random.random((n, 3)), color="blue", alpha=0.1, lines=None, )


def try_plot_faces():
    n = 10
    n1 = 20
    p = mc2.Visualizer()
    mc2.plot_faces(p=p, h=None, x=np.random.random((n, 3)),
                   faces=np.random.choice(np.arange(n), (n1, 3)), color="blue", alpha=0.3)


def try_plot_cube():
    p = mc2.Visualizer()
    limits = np.zeros((3, 2))
    limits[:, 0] = 1
    limits[:, 1] = 2

    mc2.plot_cube(p=p, h=None, limits=limits, mode="lines", color="red")
    mc2.plot_cube(p=p, h=None, limits=limits+2, mode="lines", color="white", alpha=0.5)
    mc2.plot_cube(p=p, h=None, limits=limits+4, mode="faces", color="blue", alpha=0.1)


def try_plot_spheres():

    n = 10
    p = mc2.Visualizer()
    x = np.random.random((n, 3))
    r = np.random.uniform(low=0.1, high=0.2, size=n)
    mc2.plot_spheres(p=p, h=None, x=x, r=r, color="blue", alpha=0.1)


def try_plot_bimg():
    from wzk.perlin import perlin_noise_3d
    bimg = perlin_noise_3d(shape=(256, 256, 256), res=32) < 0.3
    limits = np.zeros((3, 2))
    limits[:, 0] = 1
    limits[:, 1] = 3
    # limits += 0.5

    p = mc2.Visualizer()
    mc2.plot_bimg(p=p, h=None, img=bimg, limits=limits, color="white")


def try_arrow():
    p = mc2.Visualizer()

    p["triad"].set_object(mc2.mg.triad())
    p["triad1"].set_object(mc2.mg.triad())

    f = np.eye(4)
    mc2.plot_arrow(p=p, h=None, x=f[:3, 3], v=f[:3, 0], alpha=0.5)
    f = mc2.spatial.sample_frames()
    mc2.plot_arrow(p=p, h=None, x=f[:3, 3], v=f[:3, 0], alpha=0.5)

    p["triad"].set_transform(f)


def try_coordinate_frames(mode="A"):
    p = mc2.Visualizer()

    if mode == "A":
        mc2.plot_coordinate_frames(p=p, h=None, f=spatial.sample_frames(), color="red", scale=0.1)
        mc2.plot_coordinate_frames(p=p, h=None, f=spatial.sample_frames(), color="green", scale=0.2)
        mc2.plot_coordinate_frames(p=p, h=None, f=spatial.sample_frames(), color="blue", scale=0.3)

    elif mode == "B":
        mc2.plot_coordinate_frames(p=p, h=None, f=spatial.sample_frames(shape=5), color="blue", scale=0.1)

    elif mode == "C":
        mc2.plot_coordinate_frames(p=p, h=None, f=spatial.sample_frames(shape=2),
                                   color=[["red", "green", "blue"], ["orange", "cyan", "magenta"]], scale=0.1)


if __name__ == "__main__":
    pass

    # try_plot_points()
    # try_plot_lines()
    # try_plot_cube()
    try_plot_spheres()
    # try_plot_faces()
    # try_arrow()
    # try_plot_bimg()
    # try_coordinate_frames(mode="A")
    # try_coordinate_frames(mode="B")
    # try_coordinate_frames(mode="C")

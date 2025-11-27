import numpy as np


from wzk import spatial
from wzk.spatial.random import _sigma_theta2quaternions


def main():
    a0 = _sigma_theta2quaternions(s=0, theta1=0.1, theta2=0.1)
    a1 = _sigma_theta2quaternions(s=1, theta1=0.1, theta2=0.1)
    print("a", np.allclose(a0, a1))

    b0 = _sigma_theta2quaternions(s=0.5, theta1=0, theta2=0.1)
    b1 = _sigma_theta2quaternions(s=0.5, theta1=2*np.pi, theta2=0.1)
    print("b", np.allclose(b0, b1))

    c0 = _sigma_theta2quaternions(s=0.5, theta1=0.1, theta2=0)
    c1 = _sigma_theta2quaternions(s=0.5, theta1=0.1, theta2=2 * np.pi)
    print("c", np.allclose(c0, c1))

    spatial.random.grid_quaternions(n3=10)

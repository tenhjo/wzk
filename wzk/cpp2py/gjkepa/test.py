import numpy as np


def get_abc():
    ab = np.zeros((2, 3), dtype='f4', order='c')
    c = np.zeros((1,), dtype='f4', order='c')
    return ab, c


def hull_hull2(xa, ra, xb, rb):
    ab, c = get_abc()

    hull_hull(xa=xa.astype('f4', order='c'), ra=ra, na=len(xa),
              xb=xb.astype('f4', order='c'), rb=rb, nb=len(xb),
              ab=ab, c=c)
    return ab, c[0]


def init_ab(na, nb, d=3):
    xa = np.random.uniform(low=0, high=1, size=(na, 3)).astype('f4', order='c')
    xb = np.random.uniform(low=0, high=1, size=(nb, 3)).astype('f4', order='c')
    ra, rb = np.random.uniform(low=0.1, high=0.3, size=2)
    if d == 2:
        xa[:, 2:] = 0
        xb[:, 2:] = 0

    return (xa, ra), (xb, rb)


def test():
    (xa, ra), (xb, rb) = init_ab(na=5, nb=2)
    ab, c = hull_hull2(xa=xa, ra=ra, xb=xb, rb=rb)
    print('xa\n', xa)
    print('xb\n', xb)

    print('ab\n', ab)
    print('c\n', c)


if __name__ == '__main__':
    test()

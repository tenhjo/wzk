"""
Gradient Descent
"""
import numpy as np


def _rms(x, eps):
    return np.sqrt(x + eps)


class Optimizer:
    # Most methods come from: https://ruder.io/optimizing-gradient-descent/index.html#adam
    name = ""
    axis = -1

    def update(self, x, v):
        raise NotImplementedError


class Naive(Optimizer):
    name = "Naive"

    def __init__(self, ss=0.001):
        self.ss = ss

    def update(self, x, v):
        return -self.ss * v


class Momentum(Optimizer):
    name = "Momentum"

    def __init__(self, ss=0.001, lmbda=0.9):
        self.ss = ss
        self.lmbda = lmbda
        self.m = 0

    def update(self, x, v):
        self.m = self.lmbda * self.m + self.ss * v
        return -self.m


class NAG(Momentum):
    """
    Nesterov Accelerated Gradient
    Modified version which does not need a different location of evaluation.
    """
    name = "NAG"

    def update(self, x, v):
        self.m = self.lmbda * self.m + self.ss * v
        return -(self.lmbda * self.m + self.ss * v)


class Adagrad(Optimizer):
    name = "Adagrad"

    def __init__(self, ss=0.001, eps=1e-8):
        self.ss = ss
        self.eps = eps
        self.g = 0

    def update(self, x, v):
        self.g += v**2

        ss = self.ss / _rms(self.g, self.eps)
        return - ss * v


class Adadelta(Optimizer):
    name = "Adadelta"

    def __init__(self,  lmbda=0.9, eps=1e-8):
        self.lmbda = lmbda
        self.eps = eps
        self.e_g = 0
        self.e_x = 0

    def update(self, x, v):
        self.e_g = self.lmbda*self.e_g + (1-self.lmbda)*v**2
        self.e_x = self.lmbda*self.e_x + (1-self.lmbda)*x**2

        ss = _rms(self.e_x, self.eps) / _rms(self.e_g, self.eps)
        return - ss * v


class RMSprop(Optimizer):
    name = "RMSprop"

    def __init__(self, ss=0.001, lmbda=0.9, eps=1e-8):
        self.ss = ss
        self.lmbda = lmbda
        self.eps = eps
        self.e_g = 0

    def update(self, x, v):
        self.e_g = self.lmbda*self.e_g + (1-self.lmbda)*v**2

        ss = self.ss / _rms(self.e_g, self.eps)
        return - ss * v


class Adam(Optimizer):
    name = "Adam"

    def __init__(self, ss=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.ss = ss
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = 0
        self.v = 0
        self.beta1t = 1
        self.beta2t = 1

    def _adam(self, v):
        self.m = self.beta1 * self.m + (1-self.beta1) * v
        self.v = self.beta2 * self.v + (1-self.beta2) * v**2
        self.beta1t *= self.beta1
        self.beta2t *= self.beta2
        mt = self.m / (1-self.beta1t)
        vt = self.v / (1-self.beta2t)
        return mt, vt

    def update(self, x, v):
        mt, vt = self._adam(v=v)
        ss = self.ss / (np.sqrt(vt) + self.eps)
        return - ss * mt


class AdaMax(Adam):
    name = "AdaMax"

    def update(self, x, v):
        self.m = self.beta1 * self.m + (1-self.beta1) * v
        self.beta1t *= self.beta1
        mt = self.m / (1-self.beta1t)

        ut_a = self.beta2 * self.v
        ut_b = np.linalg.norm(v, axis=self.axis)
        ut = np.where(ut_a > ut_b, ut_a, ut_b)
        self.v = self.beta2 * self.v + (1-self.beta2) * ut_b**2
        ss = self.ss / ut
        try:
            return - ss * mt
        except ValueError:
            return - ss[..., np.newaxis] * mt


class Nadam(Adam):
    """Nesterov-accelerated Adaptive Moment Estimation"""
    name = "Nadam"

    def update(self, x, v):
        mt, vt = self._adam(v=v)

        return - self.ss / (np.sqrt(vt) + self.eps) * ((self.beta1 * mt) + (1-self.beta1) / (1-self.beta1t) * v)


class AMSGrad(Adam):

    def update(self, x, v):
        self.m = self.beta1 * self.m + (1 - self.beta1) * v
        v = self.beta2 * self.v + (1 - self.beta1) * v ** 2
        self.v = np.max(self.v, v)

        ss = self.ss / (np.sqrt(self.v) + self.eps)
        return - ss * self.m


class AdaptiveStep(Optimizer):
    name = "AdaptiveStep"

    def __init__(self, ss=0.001):
        self.ss = ss

        self.x = 0
        self.v = 0

    def update(self, x, v):
        dx = np.abs(x - self.x)
        dv = np.abs(v - self.v)
        dv2 = np.linalg.norm(dv, axis=self.axis)**2
        b0 = dv2 == 0
        dv2[b0] = 1

        ss = (dx * dv).sum(axis=self.axis) / dv2
        ss[b0] = self.ss

        if np.size(ss) > 1:
            ss = ss.reshape((ss.size,) + (1,) * (v.ndim - 1))

        self.x = x
        self.v = v

        return - ss * v


def update_x(x, p, a,
             x2, xtol, b):

    dx = a[b, np.newaxis, np.newaxis] * p[b]  # axis only one newaxis
    x2[b] = x[b] + dx
    b[b] = np.linalg.norm(dx, axis=(-2, -1)) > xtol   # noqa || axis -1
    return x2, b


class LinesearchBacktracking(Optimizer):
    def __init__(self, ss, fun=None, c=1e-1, tau=1/3, xtol=1e-5):
        self.fun = fun
        self.c = c
        self.tau = tau
        self.xtol = xtol
        self.maxiter = 20
        self.ss = ss

    def update(self, x, v):

        f = self.fun(x)
        # assert x.ndim == 2
        n = len(x)

        a = np.ones(n) * self.ss
        b = np.ones(n, dtype=bool)

        x2 = x.copy()
        f2 = f.copy()

        p = -v.copy()

        m = (v * p).sum(axis=(-2, -1))  # axis -1
        t = -self.c * m

        for i in range(self.maxiter):
            x2, b = update_x(x=x, p=p, a=a, x2=x2, xtol=self.xtol, b=b)

            if sum(b) == 0:
                break

            f2[b] = self.fun(x2[b])
            b[b] = f[b] - f2[b] < a[b] * t[b]
            # b[b] = f - f2 < a * t
            a[b] *= self.tau

            if sum(b) == 0:
                break

        return a[:, np.newaxis, np.newaxis] * p

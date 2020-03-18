import numpy as np
import timeit
from google.cloud import storage

OMEGA_MAX = 900
n = 10
N = 256


def timer(function):
    def new_function(*args, **kwargs):
        start_time = timeit.default_timer()
        val = function(*args, **kwargs)
        elapsed = timeit.default_timer() - start_time
        return val, elapsed
    return new_function


def generate_signal(t):
    A = np.random.random_sample(size=n)
    fi = np.random.random_sample(size=n) * 2 * np.math.pi
    omega = np.linspace(1, n+1, n) * OMEGA_MAX / n
    x = np.array([0 for _ in t])
    x_t = np.vectorize(lambda ti: np.sum(A * np.sin(omega * ti + fi)))
    x = x_t(t)
    return list(x)



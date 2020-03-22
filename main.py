import math
from matplotlib import pyplot as plt
import numpy as np
from lab1_signal import generate_signal, N, n, OMEGA_MAX, timer

def iexp(n):
    return complex(math.cos(n), math.sin(n))

def dft(wave):
    n = len(wave)
    return [sum((wave[k] * iexp(-2 * math.pi * i * k / n) for k in range(n)))
            for i in range(n)]

@timer
def dft_table(wave: list):
    n = len(wave)
    w_pk_arr = []
    for p in range(n):
        w_p = []
        for k in range(n):
            angle = math.degrees(2*math.pi*p*k/n)
            if angle == 0 or angle % 180 == 0:
                w_pk = math.cos(2*math.pi*p*k/n)
            elif angle % 90 == 0:
                w_pk = -1j*round(math.sin(2*math.pi*p*k/n), 2)
            else:
                w_pk = round(math.cos(2*math.pi*p*k/n), 2) - \
                        1j*round(math.sin(2*math.pi*p*k/n), 2)
            w_p.append(w_pk)
        w_pk_arr.append(w_p)
    w_pk_arr = np.array(w_pk_arr)
    return list(np.array(wave).dot(w_pk_arr))

def fft_(wave, n, start=0, stride=1):
    if n == 1: return [wave[start]]
    hn, sd = n // 2, stride * 2
    rs = fft_(wave, hn, start, sd) + fft_(wave, hn, start + stride, sd)
    for i in range(hn):
        e = iexp(-2 * math.pi * i / n)
        rs[i], rs[i + hn] = rs[i] + e * rs[i + hn], rs[i] - e * rs[i + hn]
        pass
    return rs

@timer
def fft(wave):
    assert (len(wave) % 2)==0
    return fft_(wave, len(wave))

if __name__ == "__main__":
    t = list(range(N))
    x = generate_signal(t)
    dfreq, d_elapsed = dft_table(x)
    ffreq, f_elapsed = fft(x)
    fig = plt.figure(figsize=[12, 6])
    plots = fig.subplots(3,1, sharex=True)
    plots[0].plot(t, x)
    plots[0].set_ylabel("x(t)")
    for data in [[iexp(freq * xi).real for xi in t] for freq in dfreq]:
        plots[1].plot(t, data)
    plots[1].set_ylabel(f"DFT(t)")
    for data in [[iexp(freq * xi).real for xi in t] for freq in ffreq]:
        plots[2].plot(t, data)
    plots[2].set_ylabel("FFT(t)")
    plots[2].set_xlabel(f"t (elapsed FFT: {f_elapsed}  DFT: {d_elapsed})")
    fig.show()
    plt.show()
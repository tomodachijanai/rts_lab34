import math
from matplotlib import pyplot as plt
from lab1_signal import generate_signal, N, n, OMEGA_MAX

def iexp(n):
    return complex(math.cos(n), math.sin(n))

def dft(wave):
    n = len(wave)
    return [sum((wave[k] * iexp(-2 * math.pi * i * k / n) for k in range(n)))
            for i in range(n)]

def fft_(wave, n, start=0, stride=1):
    if n == 1: return [wave[start]]
    hn, sd = n // 2, stride * 2
    rs = fft_(wave, hn, start, sd) + fft_(wave, hn, start + stride, sd)
    for i in range(hn):
        e = iexp(-2 * math.pi * i / n)
        rs[i], rs[i + hn] = rs[i] + e * rs[i + hn], rs[i] - e * rs[i + hn]
        pass
    return rs

def fft(wave):
    assert (len(wave) % 2)==0
    return fft_(wave, len(wave))

if __name__ == "__main__":
    t = list(range(N))
    x = generate_signal(t)
    dfreq = dft(x)
    ffreq = fft(x)
    fig = plt.figure(figsize=[12, 6])
    plots = fig.subplots(3,1, sharex=True)
    plots[0].plot(t, x)
    plots[0].set_ylabel("x(t)")
    for data in [[iexp(freq * xi).real for xi in t] for freq in dfreq]:
        plots[1].plot(t, data)
    plots[1].set_ylabel("DFT(t)")
    for data in [[iexp(freq * xi).real for xi in t] for freq in ffreq]:
        plots[2].plot(t, data)
    plots[2].set_ylabel("FFT(t)")
    plots[2].set_xlabel("t")
    fig.show()
    plt.show()
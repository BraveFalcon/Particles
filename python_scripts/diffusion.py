# TODO::Go to C-lib
import sys
from os import path
import matplotlib.pyplot as plt
import numpy as np
import ctypes

sys.path.append(path.abspath(path.dirname(__file__)))
import bin_parser


def get_fig_diffusion(ts, data):
    sqrs_delta_r = np.zeros(len(ts))
    lib = ctypes.CDLL(path.join(path.abspath(path.dirname(__file__)), "C-lib/diffusion.so"))
    func = lib.get_mean_sqrs_delta_r
    func.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double)
    ]
    cnum_frames = ctypes.c_int(data.shape[0])
    cnum_particles = ctypes.c_int(data.shape[1])
    cdata = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    cres = sqrs_delta_r.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    func(cnum_frames, cnum_particles, cdata, cres)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.suptitle('Diffusion', fontsize=20)
    fig.subplots_adjust(bottom=0.1, left=0.1, right=0.95, top=0.90)
    ax.set_xlabel('time, s', fontsize=14)
    ax.set_ylabel(r'$\langle \Delta r^2 \rangle$', fontsize=14)
    ax.ticklabel_format(style='sci', scilimits=(-1, 2))

    ax.plot(ts, sqrs_delta_r, 'ro', ms=2)

    return fig


if __name__ == '__main__':
    experiment_path = sys.argv[1]

    data, energies, ts = bin_parser.read_file(path.join(experiment_path, 'data.bin'))
    fig = get_fig_diffusion(ts, data)
    plt.show()

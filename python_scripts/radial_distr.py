import sys
from os import path

import matplotlib.pyplot as plt
import numpy as np
import ctypes

sys.path.append(path.abspath(path.dirname(__file__)))
import bin_parser


def get_fig_rad_distr(data, cell_size, cut_dist=-1.0):
    if cut_dist < 0:
        cut_dist = cell_size / 2
    num_bins = int(data.shape[1] * (cut_dist / cell_size) ** 3 * np.sqrt(data.shape[0]))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.92)
    fig.suptitle("Radial distribution", fontsize=20)
    ax.set_xlabel(r"$r, \sigma$", fontsize=14)
    ax.set_ylabel(r"$g(r)$", fontsize=14)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax.set_xlim((0, cut_dist))
    ax.grid(True)

    hs = np.zeros(num_bins, dtype='float64')
    bins = np.linspace(0, cut_dist, num_bins, endpoint=False)

    lib = ctypes.CDLL(path.join(path.abspath(path.dirname(__file__)), "C-lib/rad.so"))
    func = lib.get_distr
    func.argtypes = [ctypes.POINTER(ctypes.c_double),
                     ctypes.c_int,
                     ctypes.c_int,
                     ctypes.c_double,
                     ctypes.POINTER(ctypes.c_double),
                     ctypes.c_int,
                     ctypes.c_double
                     ]
    cdata = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    cnum_frames = ctypes.c_int(data.shape[0])
    cnum_particles = ctypes.c_int(data.shape[1])
    ccell_size = ctypes.c_double(cell_size)
    cres = hs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    cnum_bins = ctypes.c_int(num_bins)
    cmax_r = ctypes.c_double(cut_dist)
    func(cdata, cnum_frames, cnum_particles, ccell_size, cres, cnum_bins, cmax_r)

    ax.bar(bins, hs, align='edge', width=cut_dist / num_bins)
    return fig


if __name__ == '__main__':
    experiment_path = sys.argv[1]
    if len(sys.argv) == 2:
        sys.argv.append(-1)

    data, energies, ts = bin_parser.read_file(path.join(experiment_path, 'data.bin'))
    cell_size = bin_parser.parse_info_file(path.join(experiment_path, 'info.txt'))['CELL_SIZE']

    fig = get_fig_rad_distr(data, cell_size, float(sys.argv[2]))
    plt.show()

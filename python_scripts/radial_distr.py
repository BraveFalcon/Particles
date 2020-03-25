import sys
from os import path

import matplotlib.pyplot as plt
import numpy as np
import ctypes

sys.path.append(path.split(path.split(path.abspath(__file__))[0])[0])
from python_scripts.experiment import Experiment
from python_scripts import CLib


def get_figure(exp, cut_dist=-1.0):
    if cut_dist < 0:
        cut_dist = exp.cell_size / 2
    num_bins = min(1000, int(np.sqrt(
        exp.num_frames * exp.num_particles * (4 / 3 * np.pi * exp.num_particles * (cut_dist / exp.cell_size) ** 3 - 1)
    )))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.subplots_adjust(bottom=0.1, left=0.1, right=0.95, top=0.92)
    fig.suptitle("Radial distribution", fontsize=20)
    ax.set_xlabel(r"$r, \sigma$", fontsize=14)
    ax.set_ylabel(r"$g(r)$", fontsize=14)
    ax.set_xlim((0, cut_dist))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.grid(True, 'major', linewidth=1.5)
    ax.grid(True, 'minor', linewidth=0.4)

    bins, hs = CLib.radial_distr(exp, cut_dist, num_bins)

    ax.plot(bins, hs, 'b', label='MD simulation')
    xs = np.linspace(1e-10, cut_dist, 1000)
    approx_line = np.exp(-4 * (xs ** -12 - xs ** -6) / exp.temperature)
    # ax.plot(xs, approx_line, 'r--', lw=1.2, label='Boltzmann distribution')

    ax.legend()
    return fig


if __name__ == '__main__':
    experiment_path = sys.argv[1]
    if len(sys.argv) == 2:
        sys.argv.append(-1)

    experiment = Experiment(experiment_path)

    fig = get_figure(experiment, float(sys.argv[2]))
    plt.show()

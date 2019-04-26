import sys
from os import path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

sys.path.append(path.split(path.split(path.abspath(__file__))[0])[0])
from python_scripts import bin_parser


def get_fig_momentum(ts, data):
    moms = np.mean(data[:, :, 1, 0], 1)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Momentum fluctuations', fontsize=20)
    fig.subplots_adjust(bottom=0.1, left=0.1, right=0.95, top=0.90)
    axs[0].set_xlabel('time, s', fontsize=14)
    axs[0].set_ylabel(r'$\Delta p$', fontsize=14)
    axs[0].ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))
    axs[0].ticklabel_format(axis='x', style='sci', scilimits=(-2, 3))
    axs[1].set_xlabel(r'$\Delta p$', fontsize=14)
    axs[1].ticklabel_format(axis='x', style='sci', scilimits=(-1, 1))
    axs[1].ticklabel_format(axis='y', style='sci', scilimits=(-2, 1))
    axs[1].grid(True, axis='y')

    mean_mom = np.mean(moms)
    std_mom = np.std(moms)
    deltas = moms - mean_mom

    axs[0].plot(ts, deltas, 'ro', ms=2)
    axs[0].plot([ts[0], ts[-1]], [std_mom] * 2, 'g-', lw=2)
    axs[0].plot([ts[0], ts[-1]], [-std_mom] * 2, 'g-', lw=2)

    axs[1].hist(deltas, bins=30, density=True)
    xs = np.linspace(np.min(deltas), np.max(deltas), 200)
    norm_distr = stats.norm.pdf(xs, 0, std_mom)
    axs[1].plot(xs, norm_distr, 'r', lw=1.5)

    return fig


if __name__ == "__main__":
    experiment_path = sys.argv[1]

    data, energies, ts = bin_parser.read_file(path.join(experiment_path, 'data.bin'))

    fig = get_fig_momentum(ts, data)
    plt.show()

import sys
from os import path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def get_fig_energy(ts, energies):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Energy fluctuations', fontsize=20)
    fig.subplots_adjust(bottom=0.1, left=0.1, right=0.95, top=0.90)
    axs[0].set_xlabel('time, s', fontsize=14)
    axs[0].set_ylabel(r'$\Delta E / E_0$', fontsize=14)
    axs[0].ticklabel_format(style='sci', scilimits=(-1, 1))
    axs[1].set_xlabel(r'$\Delta E / E_0$', fontsize=14)
    axs[1].ticklabel_format(axis='x', style='sci', scilimits=(-1, 1))
    axs[1].ticklabel_format(axis='y', style='sci', scilimits=(-2, 1))
    axs[1].grid(True, axis='y')

    mean_energy = np.mean(energies)
    std_energy = np.std(energies)
    relative_deltas = energies / mean_energy - 1

    axs[0].plot(ts, relative_deltas, 'ro', ms=2)
    axs[0].plot([ts[0], ts[-1]], [std_energy / mean_energy] * 2, 'g-', lw=2)
    axs[0].plot([ts[0], ts[-1]], [-std_energy / mean_energy] * 2, 'g-', lw=2)

    axs[1].hist(relative_deltas, bins='auto', density=True)
    xs = np.linspace(np.min(relative_deltas), np.max(relative_deltas), 200)
    norm_distr = stats.norm.pdf(xs, 0, std_energy / mean_energy)
    axs[1].plot(xs, norm_distr, 'r', lw=1.5)

    return fig


if __name__ == '__main__':
    experiment_path = sys.argv[1]

    ts = np.load(path.join(experiment_path, 'data', "ts.npy"))
    energies = np.load(path.join(experiment_path, 'data', "energies.npy"))

    fig = get_fig_energy(ts, energies)
    plt.show()

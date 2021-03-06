import sys
from os import path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

sys.path.append(path.split(path.split(path.abspath(__file__))[0])[0])
from python_scripts.experiment import Experiment


def get_figure(exp):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Energy fluctuations', fontsize=20)
    fig.subplots_adjust(bottom=0.1, left=0.1, right=0.95, top=0.90)
    axs[0].set_xlabel('time, $\sigma \sqrt{m / \epsilon}$', fontsize=14)
    axs[0].set_ylabel(r'$\Delta E / E_0$', fontsize=14)
    axs[0].ticklabel_format(axis='x', style='sci', scilimits=(-2, 3))
    axs[0].ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))
    axs[1].set_xlabel(r'$\Delta E / E_0$', fontsize=14)
    axs[1].ticklabel_format(axis='x', style='sci', scilimits=(-1, 1))
    axs[1].set_yticks([])
    axs[1].grid(True, axis='x')

    start_energy = exp.energies[0]
    std_energy = np.std(exp.energies)
    relative_deltas = exp.energies / start_energy - 1

    axs[0].plot(exp.ts, relative_deltas, 'ro', ms=2)
    axs[0].plot([exp.ts[0], exp.ts[-1]], [std_energy / start_energy] * 2, 'g-', lw=2)
    axs[0].plot([exp.ts[0], exp.ts[-1]], [-std_energy / start_energy] * 2, 'g-', lw=2)

    axs[1].hist(relative_deltas, bins='auto', density=True)
    xs = np.linspace(np.min(relative_deltas), np.max(relative_deltas), 200)
    norm_distr = stats.norm.pdf(xs, 0, std_energy / start_energy)
    axs[1].plot(xs, norm_distr, 'r', lw=1.5)

    return fig


if __name__ == '__main__':
    experiment_path = sys.argv[1]

    experiment = Experiment(experiment_path)

    fig = get_figure(experiment)
    plt.show()

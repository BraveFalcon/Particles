import sys
from os import path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

sys.path.append(path.split(path.split(path.abspath(__file__))[0])[0])
from python_scripts.experiment import Experiment


def get_figure(exp):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.suptitle('...', fontsize=20)
    fig.subplots_adjust(bottom=0.1, left=0.1, right=0.95, top=0.90)
    ax.set_xlabel('time, $\sigma \sqrt{m / \epsilon}$', fontsize=14)
    ax.set_ylabel(r'$Energy, \epsilon$', fontsize=14)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-1, 3))

    kinetic = 0.5 * np.sum(np.linalg.norm(exp.vels, axis=2) ** 2, axis=1)
    potensional = exp.energies - kinetic

    ax.plot(exp.ts, exp.energies, 'k', label='Full')
    ax.plot(exp.ts, kinetic, 'r', label='Kinetic')
    ax.plot(exp.ts, potensional, 'b', label='Potens')

    ax.legend()

    return fig


if __name__ == '__main__':
    experiment_path = sys.argv[1]

    experiment = Experiment(experiment_path)

    fig = get_figure(experiment)
    plt.show()

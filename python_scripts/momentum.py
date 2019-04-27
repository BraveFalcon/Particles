import sys
from os import path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

sys.path.append(path.split(path.split(path.abspath(__file__))[0])[0])
from python_scripts.experiment import Experiment


def get_figure(exp):
    moms = np.sum(exp.vels[:, :, 0], axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.suptitle('Momentum fluctuations', fontsize=20)
    fig.subplots_adjust(bottom=0.1, left=0.11, right=0.95, top=0.90)
    ax.set_xlabel('time, $\sigma \sqrt{m / \epsilon}$', fontsize=14)
    ax.set_ylabel(r'$p_x, \sqrt{m \epsilon}$', fontsize=14)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 3))

    std_mom = np.std(moms)

    ax.plot(exp.ts, moms, 'ro', ms=2)
    ax.plot([exp.ts[0], exp.ts[-1]], [std_mom] * 2, 'g-', lw=2)
    ax.plot([exp.ts[0], exp.ts[-1]], [-std_mom] * 2, 'g-', lw=2)

    return fig


if __name__ == "__main__":
    experiment_path = sys.argv[1]

    experiment = Experiment(experiment_path)

    fig = get_figure(experiment)
    plt.show()

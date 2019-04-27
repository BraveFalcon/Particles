import sys
from os import path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(path.split(path.split(path.abspath(__file__))[0])[0])
from python_scripts.experiment import Experiment


def get_figure(exp):
    temperatures = 1 / 3 * np.mean(np.linalg.norm(exp.vels, axis=2) ** 2, axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.suptitle('Temperature stabilization and fluctuations', fontsize=20)
    fig.subplots_adjust(bottom=0.1, left=0.1, right=0.95, top=0.90)
    ax.set_xlabel('time, $\sigma \sqrt{m / \epsilon}$', fontsize=14)
    ax.set_ylabel(r'$T, \epsilon$', fontsize=14)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-1, 3))

    mean_temp = np.mean(temperatures)
    std_temp = np.std(temperatures)

    ax.plot(exp.ts, temperatures, 'ro', ms=2)
    ax.plot([exp.ts[0], exp.ts[-1]], [mean_temp + std_temp] * 2, 'g-', lw=2)
    ax.plot([exp.ts[0], exp.ts[-1]], [mean_temp - std_temp] * 2, 'g-', lw=2)

    return fig


if __name__ == '__main__':
    experiment_path = sys.argv[1]

    experiment = Experiment(experiment_path)

    fig = get_figure(experiment)
    plt.show()

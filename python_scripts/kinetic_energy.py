import sys
from os import path

import matplotlib.pyplot as plt
import numpy as np


def get_fig_kin_energy(ts, kin_energies):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.suptitle('Kinetic energy stabilization and fluctuations', fontsize=20)
    fig.subplots_adjust(bottom=0.1, left=0.1, right=0.95, top=0.90)
    ax.set_xlabel('time, s', fontsize=14)
    ax.set_ylabel(r'$K / K_{mean}$', fontsize=14)
    ax.ticklabel_format(style='sci', scilimits=(-1, 2))

    mean_kin_energy = np.mean(kin_energies[len(kin_energies) // 2:])
    std_kin_energy = np.std(kin_energies[len(kin_energies) // 2:])

    ax.plot(ts, kin_energies / mean_kin_energy, 'ro', ms=2)
    ax.plot([ts[0], ts[-1]], [1 + std_kin_energy / mean_kin_energy] * 2, 'g-', lw=2)
    ax.plot([ts[0], ts[-1]], [1 - std_kin_energy / mean_kin_energy] * 2, 'g-', lw=2)

    return fig


if __name__ == '__main__':
    experiment_path = sys.argv[1]
    ts = np.load(path.join(experiment_path, 'data', "ts.npy"))
    data = np.load(path.join(experiment_path, 'data', "data.npy"))
    kin_energies = np.mean(np.linalg.norm(data[:, :, 1], axis=2) ** 2, axis=1)
    fig = get_fig_kin_energy(ts, kin_energies)
    plt.show()

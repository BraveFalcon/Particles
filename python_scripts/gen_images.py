import sys
from os import path

import matplotlib.pyplot as plt
import numpy as np

__import__('sys').path.append(path.abspath(path.dirname(__file__)))
import energy, vels, momentum, kinetic_energy


def gen_images(experiment_path):
    ts = np.load(path.join(experiment_path, 'data', "ts.npy"))
    energies = np.load(path.join(experiment_path, 'data', "energies.npy"))
    data = np.load(path.join(experiment_path, 'data', 'data.npy'))

    fig = vels.get_fig_vels(data)
    plt.savefig(path.join(experiment_path, 'images', 'vel_distr.svg'), dpi=500)
    plt.close(fig)

    fig = energy.get_fig_energy(ts, energies)
    plt.savefig(path.join(experiment_path, 'images', 'energy.svg'))
    plt.close(fig)

    fig = momentum.get_fig_momentum(ts, data)
    plt.savefig(path.join(experiment_path, 'images', 'momentum.svg'))
    plt.close(fig)

    fig = kinetic_energy.get_fig_kin_energy(ts, data)
    plt.savefig(path.join(experiment_path, 'images', 'kin_energy.svg'))
    plt.close(fig)


if __name__ == "__main__":
    experiment_path = sys.argv[1]

    gen_images(experiment_path)

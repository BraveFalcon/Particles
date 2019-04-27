import sys
from os import path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(path.split(path.split(path.abspath(__file__))[0])[0])
from python_scripts.experiment import Experiment


def get_figure(exp):
    x_vels = exp.vels.ravel()
    modules_vel = np.linalg.norm(exp.vels, axis=2).ravel()

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(bottom=0.1, left=0.05, right=0.95, top=0.92)
    fig.suptitle("Velocity distribution", fontsize=20)

    axs[0].set_xlabel(r"$v_i$, $\sqrt{\epsilon / m}$", fontsize=14)
    axs[0].ticklabel_format(style='sci', scilimits=(-2, 2))
    axs[0].set_yticks([])
    axs[0].set_xlim(np.min(x_vels), np.max(x_vels))
    axs[0].grid(True, axis='x')

    axs[1].set_xlabel(r"$|v|$, $\sqrt{\epsilon / m}$", fontsize=14)
    axs[1].ticklabel_format(style='sci', scilimits=(-2, 2))
    axs[1].set_yticks([])
    axs[1].set_xlim(0, np.max(modules_vel))
    axs[1].grid(True, axis='x')

    axs[0].hist(x_vels, bins=int(np.sqrt(len(x_vels))), density=True)
    xs = np.linspace(np.min(x_vels), np.max(x_vels), 1000)
    norm_distr = np.sqrt(1 / (2 * np.pi * exp.temperature)) * np.exp(-xs ** 2 / (2 * exp.temperature))
    axs[0].plot(xs, norm_distr, 'r--', lw=1.2)

    axs[1].hist(modules_vel, bins=int(np.sqrt(len(modules_vel))), density=True)
    xs = np.linspace(0, np.max(modules_vel), 1000)
    maxw_distr = 4 * np.pi * xs ** 2 * np.power(1 / (2 * np.pi * exp.temperature), 3 / 2) * np.exp(
        - xs ** 2 / (2 * exp.temperature))
    axs[1].plot(xs, maxw_distr, 'r--', lw=1.2)

    return fig


if __name__ == "__main__":
    experiment_path = sys.argv[1]

    experiment = Experiment(experiment_path)

    fig = get_figure(experiment)
    plt.show()

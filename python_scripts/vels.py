import sys
from os import path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(path.abspath(path.dirname(__file__)))
import bin_parser


def get_fig_vels(data, temperature):
    x_vels = data[:, :, 1].ravel()
    modules_vel = np.linalg.norm(data[:, :, 1], axis=2).ravel()

    fig, axs = plt.subplots(1, 2, figsize=(18, 9))
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
    norm_distr = np.sqrt(1 / (2 * np.pi * temperature)) * np.exp(-xs ** 2 / (2 * temperature))
    axs[0].plot(xs, norm_distr, 'r--', lw=1.2)

    axs[1].hist(modules_vel, bins=int(np.sqrt(len(modules_vel))), density=True)
    xs = np.linspace(0, np.max(modules_vel), 1000)
    maxw_distr = 4 * np.pi * xs ** 2 * np.power(1 / (2 * np.pi * temperature), 3 / 2) * np.exp(
        - xs ** 2 / (2 * temperature))
    axs[1].plot(xs, maxw_distr, 'r--', lw=1.2)

    return fig


if __name__ == "__main__":
    experiment_path = sys.argv[1]

    data, energies, ts = bin_parser.read_file(path.join(experiment_path, 'data.bin'))
    temperature = bin_parser.parse_results_file(path.join(experiment_path, 'results.txt'))['TEMPERATURE'][0]
    fig = get_fig_vels(data, temperature)
    plt.show()

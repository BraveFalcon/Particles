import sys
from os import path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

sys.path.append(path.abspath(path.dirname(__file__)))
import bin_parser


def get_fig_vels(data):
    vels = data[:, :, 1, 0].ravel()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.92)
    fig.suptitle("Velocity distribution", fontsize=20)
    ax.set_xlabel(r"$v_x$, m/s", fontsize=14)
    ax.ticklabel_format(style='sci', scilimits=(-2, 2))
    ax.grid(True, axis='y')

    ax.hist(vels, bins='auto', density=True)
    xs = np.linspace(np.min(vels), np.max(vels), 200)
    norm_distr = stats.norm.pdf(xs, np.mean(vels), np.std(vels))
    ax.plot(xs, norm_distr, 'r', lw=1.5)

    return fig


if __name__ == "__main__":
    experiment_path = sys.argv[1]

    data, energies, ts = bin_parser.read_file(path.join(experiment_path, 'data.bin'))

    fig = get_fig_vels(data)
    plt.show()

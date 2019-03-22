import sys
from os import path
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(path.abspath(path.dirname(__file__)))
import bin_parser


def get_fig_diffusion(ts, data):
    sqrs_delta_r = np.mean(np.linalg.norm(data[:, :, 0] - data[0, :, 0], axis=2) ** 2, axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.suptitle('Diffusion', fontsize=20)
    fig.subplots_adjust(bottom=0.1, left=0.1, right=0.95, top=0.90)
    ax.set_xlabel('time, s', fontsize=14)
    ax.set_ylabel(r'$\langle \Delta r^2 \rangle$', fontsize=14)
    ax.ticklabel_format(style='sci', scilimits=(-1, 2))

    ax.plot(ts, sqrs_delta_r, 'ro', ms=2)

    return fig


if __name__ == '__main__':
    experiment_path = sys.argv[1]

    data, energies, ts = bin_parser.read_file(path.join(experiment_path, 'data.bin'))

    fig = get_fig_diffusion(ts, data)
    plt.show()

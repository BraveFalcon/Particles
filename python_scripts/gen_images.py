import sys
from os import path
import matplotlib.pyplot as plt

sys.path.append(path.abspath(path.dirname(__file__)))
import energy, vels, momentum, kinetic_energy, diffusion, bin_parser, radial_distr


def gen_images(experiment_path):
    data, energies, ts = bin_parser.read_file(path.join(experiment_path, "data.bin"))
    info = bin_parser.parse_info_file(path.join(experiment_path, 'info.txt'))

    fig = vels.get_fig_vels(data)
    plt.savefig(path.join(experiment_path, 'images', 'vel_distr.eps'))
    plt.close(fig)

    fig = energy.get_fig_energy(ts, energies)
    plt.savefig(path.join(experiment_path, 'images', 'energy.eps'))
    plt.close(fig)

    fig = momentum.get_fig_momentum(ts, data)
    plt.savefig(path.join(experiment_path, 'images', 'momentum.eps'))
    plt.close(fig)

    fig = kinetic_energy.get_fig_kin_energy(ts, data)
    plt.savefig(path.join(experiment_path, 'images', 'kin_energy.eps'))
    plt.close(fig)

    fig = diffusion.get_fig_diffusion(ts, data)
    plt.savefig(path.join(experiment_path, 'images', 'diffusion.eps'))
    plt.close(fig)

    fig = radial_distr.get_fig_rad_distr(data, info['CELL_SIZE'])
    plt.savefig(path.join(experiment_path, 'images', 'radial_distr.eps'))
    plt.close(fig)


if __name__ == "__main__":
    experiment_path = sys.argv[1]

    gen_images(experiment_path)

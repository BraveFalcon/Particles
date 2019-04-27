import sys
from os import path
import matplotlib.pyplot as plt

sys.path.append(path.split(path.split(path.abspath(__file__))[0])[0])
from python_scripts import energy, vels, momentum, temperature, diffusion, radial_distr
from python_scripts.experiment import Experiment


def gen_images(experiment_path):
    experimentt = Experiment(experiment_path)

    image_format = "png"
    dpi = 400

    fig = vels.get_figure(experimentt)
    plt.savefig(path.join(experiment_path, 'images', 'vel_distr.' + image_format), dpi=dpi)
    plt.close(fig)

    fig = energy.get_figure(experimentt)
    plt.savefig(path.join(experiment_path, 'images', 'energy.' + image_format), dpi=dpi)
    plt.close(fig)

    fig = momentum.get_figure(experimentt)
    plt.savefig(path.join(experiment_path, 'images', 'momentum.' + image_format), dpi=dpi)
    plt.close(fig)

    fig = temperature.get_figure(experimentt)
    plt.savefig(path.join(experiment_path, 'images', 'temperature.' + image_format), dpi=dpi)
    plt.close(fig)

    fig = diffusion.get_figure(experimentt)
    plt.savefig(path.join(experiment_path, 'images', 'diffusion.' + image_format), dpi=dpi)
    plt.close(fig)

    fig = radial_distr.get_figure(experimentt)
    plt.savefig(path.join(experiment_path, 'images', 'radial_distr.' + image_format), dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":
    experiment_path = sys.argv[1]

    gen_images(experiment_path)

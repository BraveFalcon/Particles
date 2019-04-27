import sys
from os import path
import os
import matplotlib.pyplot as plt

sys.path.append(path.split(path.split(path.abspath(__file__))[0])[0])
from python_scripts import energy, vels, momentum, temperature, diffusion, radial_distr
from python_scripts.experiment import Experiment


def gen_images(experiment, images_path):
    image_format = "png"
    dpi = 400

    fig = vels.get_figure(experiment)
    plt.savefig(path.join(images_path, 'vel_distr.' + image_format), dpi=dpi)
    plt.close(fig)

    fig = energy.get_figure(experiment)
    plt.savefig(path.join(images_path, 'energy.' + image_format), dpi=dpi)
    plt.close(fig)

    fig = momentum.get_figure(experiment)
    plt.savefig(path.join(images_path, 'momentum.' + image_format), dpi=dpi)
    plt.close(fig)

    fig = temperature.get_figure(experiment)
    plt.savefig(path.join(images_path, 'temperature.' + image_format), dpi=dpi)
    plt.close(fig)

    fig = diffusion.get_figure(experiment)
    plt.savefig(path.join(images_path, 'diffusion.' + image_format), dpi=dpi)
    plt.close(fig)

    fig = radial_distr.get_figure(experiment)
    plt.savefig(path.join(images_path, 'radial_distr.' + image_format), dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":
    experiment_path = sys.argv[1]
    if not path.exists(path.join(experiment_path, 'images')):
        os.mkdir(path.join(experiment_path, 'images'))

    gen_images(Experiment(experiment_path), path.join(experiment_path, 'images'))

import os
from os import path
import sys
import numpy as np

sys.path.append(path.split(path.split(path.abspath(__file__))[0])[0])
from python_scripts.experiment import Experiment
from python_scripts import gen_images, gen_results

if __name__ == "__main__":
    experiment_path = sys.argv[1]
    start_time = float(sys.argv[2])

    exp = Experiment(experiment_path)
    exp.cut(slice(np.argmax(exp.ts > start_time), exp.num_frames + 1))
    exp.save_pickle(experiment_path)

    os.mkdir(path.join(experiment_path, 'images (cut)'))
    gen_images.gen_images(exp, path.join(experiment_path, 'images (cut)'))
    gen_results.gen_results(exp, experiment_path)

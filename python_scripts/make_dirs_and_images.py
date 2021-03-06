import os
import os.path as path
import shutil
import stat
import sys
from datetime import datetime

sys.path.append(path.split(path.split(path.abspath(__file__))[0])[0])
from python_scripts import gen_images, gen_results
from python_scripts.experiment import Experiment

print("Replacing files...")
c_path = sys.argv[1]
if path.exists(path.join(c_path, 'data.bin')):
    experiments_path = sys.argv[2]
    new_dir_name = datetime.strftime(datetime.now(), "%y.%m.%d-%H:%M:%S")
    os.mkdir(path.join(experiments_path, new_dir_name))
    shutil.move(path.join(c_path, 'data.bin'), path.join(experiments_path, new_dir_name, 'data.bin'))
    shutil.move(path.join(c_path, 'log.txt'), path.join(experiments_path, new_dir_name, 'log.txt'))
    os.chdir(path.join(experiments_path, new_dir_name))

print("Importing data...")
experiment = Experiment('.')

print("Generating results...")
gen_results.gen_results(experiment, ".")

print("Generating images...")
os.mkdir("images")
gen_images.gen_images(experiment, 'images')

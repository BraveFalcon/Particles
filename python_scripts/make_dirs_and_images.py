import os
import os.path as path
import shutil
import stat
import sys
from datetime import datetime

sys.path.append(path.abspath(path.dirname(__file__)))
import load_data, gen_images

data_path = sys.argv[1]
experiments_path = sys.argv[2]
new_dir_name = datetime.strftime(datetime.now(), "%m.%d.%y_%H:%M")

os.makedirs(path.join(experiments_path, new_dir_name, 'data'))
os.makedirs(path.join(experiments_path, new_dir_name, 'images'))

print("Coping files...")
shutil.copy(path.join(data_path, 'model_constants.h'), path.join(experiments_path, new_dir_name, 'info.txt'))
shutil.move(path.join(data_path, 'data.xyz'), path.join(experiments_path, new_dir_name, 'data', 'data.xyz'))

os.chdir(path.join(experiments_path, new_dir_name))

print("Generating numpy cash...")
load_data.resave_xyz_to_npy(path.join('data', 'data.xyz'), 'data')
print("Generating images...")
gen_images.gen_images('.')

for file in os.listdir('data'):
    os.chmod(path.join('data', file), stat.S_IREAD)
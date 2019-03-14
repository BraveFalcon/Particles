import os
import os.path as path
import shutil
import stat
from datetime import datetime

__import__('sys').path.append(path.abspath(path.dirname(__file__)))
import load_data, gen_images

new_dir_name = datetime.strftime(datetime.now(), "%m.%d.%y_%H:%M")

os.chdir('..')
os.makedirs(path.join('experiments', new_dir_name, 'data'))
os.makedirs(path.join('experiments', new_dir_name, 'images'))

print("Coping files...")
shutil.move(path.join('cmake-build-debug', 'info.txt'), path.join("experiments", new_dir_name, 'info.txt'))
shutil.move(path.join('cmake-build-debug', 'data.xyz'), path.join("experiments", new_dir_name, 'data', 'data.xyz'))

os.chdir(path.join('experiments', new_dir_name))

print("Generating numpy cash...")
load_data.resave_xyz_to_npy(path.join('data', 'data.xyz'), 'data')
print("Generating images...")
gen_images.gen_images('.')

for file in os.listdir('data'):
    os.chmod(path.join('data', file), stat.S_IREAD)

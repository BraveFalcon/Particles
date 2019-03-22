import os
import os.path as path
import shutil
import stat
import sys
from datetime import datetime

sys.path.append(path.abspath(path.dirname(__file__)))
import gen_images

data_path = sys.argv[1]
experiments_path = sys.argv[2]
new_dir_name = datetime.strftime(datetime.now(), "%m.%d.%y_%H:%M")

os.makedirs(path.join(experiments_path, new_dir_name, 'images'))

print("Replacing files...")
shutil.copy(path.join(data_path, 'info.txt'), path.join(experiments_path, new_dir_name, 'info.txt'))
shutil.move(path.join(data_path, 'data.bin'), path.join(experiments_path, new_dir_name, 'data.bin'))

os.chdir(path.join(experiments_path, new_dir_name))
os.chmod('data.bin', stat.S_IREAD)
os.chmod('info.txt', stat.S_IREAD)

print("Generating images...")
gen_images.gen_images('.')

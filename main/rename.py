# This has the simple purpose to rename the delivered FERET Files from names such as 
"00002_940928_fa.png"
# todo
"0.jpg"
from natsort import natsorted
# natsorted(files, alg=ns.IGNORECASE)
import os
# path = '/Users/janheimes/Main/DTU/02238_Biometric_systems/Recognition_Anonymization_Face/FERET/reference'
path = '/Users/janheimes/Main/DTU/02238_Biometric_systems/results/blur/20'

files = os.listdir(path)
# we have file names such as 00090_931230_fb
# But they are annoying to interate through
# Hence renaming to 12 , 13, 14 etc
extension = '.jpg'

os.chdir(path)
for i, filename in enumerate(natsorted(files)):
    os.rename(src=filename, dst='{}{}'.format(i, extension))


# # for index,file in zip(natsorted(os.listdir(path), key=lambda x: int(x.split('.')[0]), alg=ns.IGNORECASE), L):
# 
#     # print(file.split('_')[0].strip('0'))
#     # new_name = file.split('_')[0].strip('0')
#     os.rename(os.path.join(path, file), os.path.join(path, ''.join([str(index), '.jpg'])))
#     print("succsess")
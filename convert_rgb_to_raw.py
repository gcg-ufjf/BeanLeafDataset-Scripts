from PIL import Image
from tqdm import tqdm
import numpy as np

import os

# palette (color map) describes the (R, G, B): Label pair
palette = {(0,   0,   0) : 0 ,
         (255, 0, 0) : 1, #leaf
         (0, 255, 0) : 2 #marker
         }

def convert_from_color_segmentation(arr_3d):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i
    return arr_2d


label = '../'
new = '../'

for i in range(1, 613):
    leafNumber = i
    print( str(leafNumber).zfill(3))

    label_dir = label + str(leafNumber).zfill(3) + '/segmentation/'
    label_files = os.listdir(label_dir)

    for l_f in tqdm(label_files):
        try:
            raw_path = os.path.join(label_dir, l_f.replace(".png", ".raw"))
            os.remove(raw_path)
        except:
            pass

    for l_f in tqdm(label_files):
        l_f_path = os.path.join(label_dir, l_f.replace(".raw", ".png"))
        print('>>>>>', l_f_path)
        arr = np.array(Image.open(l_f_path))
        arr = arr[:,:,0:3]
        arr_2d = convert_from_color_segmentation(arr)
        raw_path = os.path.join(label_dir, l_f.replace(".png", ".raw"))
        print(arr_2d.shape)
        print(arr_2d[ arr_2d != 0])
        arr_2d.tofile(raw_path)
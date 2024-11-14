import os
import numpy as np
from matplotlib import pylab as plt
from PIL import Image
from area_dict import *
from trainval_imgdict import *

np.set_printoptions(precision=15)

img_folder_path = '../leafRaw/'

# Loop through each file in the folder
for filename in os.listdir(img_folder_path):
  try:
      print('>>> ', filename)
      leaf = imgs_dict[filename]
      leaf_area = area_dict[leaf]
      leaf_area = np.double(leaf_area)
    
      file_path = img_folder_path + filename
      mask = Image.open(file_path)
      np_leaf = np.asarray(mask, dtype=np.double)
      n_pixels_leaf = np.count_nonzero(np_leaf == 1)
      np_leaf = np_leaf * leaf_area/n_pixels_leaf
      
      raw_file_path = (file_path.replace('leafRaw', 'leafRaw2')).replace('.png', '.raw')    
      np_leaf.tofile(raw_file_path)
      
      np_background = np.zeros((512,512), dtype=np.double)
      raw_file_path = (file_path.replace('leafRaw', 'background')).replace('.png', '.raw')   
      np_background.tofile(raw_file_path)
      
  except:
    pass

from tqdm import tqdm
from glob import glob
from os import path
import numpy as np
import os


def plot_colormaps(marker_raw, leaf_raw, merged_raw):
    import matplotlib.pyplot as plt

    image_shape = (512, 512)

    cmap = plt.cm.jet
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(10, 5))

    im = ax1.imshow(marker_raw.reshape(image_shape), cmap=cmap)
    ax1.figure.colorbar(im, ax=ax1)
    ax1.set_title('Colormap Marker Raw')
    ax1.axis('off')

    im = ax2.imshow(leaf_raw.reshape(image_shape), cmap=cmap)
    ax2.figure.colorbar(im, ax=ax2)
    ax2.set_title('Colormap Leaf Raw')
    ax2.axis('off')

    im = ax3.imshow(merged_raw.reshape(image_shape), cmap=cmap)
    ax3.figure.colorbar(im, ax=ax3)
    ax3.set_title('Colormap Merged Raw')
    ax3.axis('off')

    fig_path = path.join('../', merged_name.replace(".raw", ".png"))
    fig.savefig(fig_path, dpi=300)


orig = '../'
output = '../'


for i in range(1, 613):
    leafNumber = i
    print( str(leafNumber).zfill(3))

    marker_area_dir = orig + str(leafNumber).zfill(3) + '/area_estimation/marker/'
    leaf_area_dir = orig + str(leafNumber).zfill(3) + '/area_estimation/leaf/'
    output_dir =  output + str(leafNumber).zfill(3) + '/area_estimation/'

    if not path.isdir(output_dir):
        print("Creating folder: ", output_dir)
        os.makedirs(output_dir)
    else:
        print("Output folder already exists:", output_dir)


    marker_raw_paths = glob(path.join(marker_area_dir, "*.raw"))
    leaf_raw_paths = glob(path.join(leaf_area_dir, "*.raw"))

    marker_raw_paths.sort()
    leaf_raw_paths.sort()

    for marker_path, leaf_path in tqdm(zip(marker_raw_paths, leaf_raw_paths)):
        marker_name = path.basename(marker_path)
        leaf_name = path.basename(leaf_path)

        if marker_name != leaf_name:
            print(f"Marker '{marker_name}' doesn't match leaf '{leaf_name}'.")
            print("Check if the files in the marker and leaf folders match each other.")
            #break

        marker_raw = np.fromfile(marker_path, np.double) * 1000
        leaf_raw = np.fromfile(leaf_path, np.double) * 1000

        merged_name = marker_name
        merged_raw = (marker_raw + leaf_raw).astype(np.float32)
        merged_raw.tofile(path.join(output_dir, merged_name))

        #plot_colormaps(marker_raw, leaf_raw, merged_raw)

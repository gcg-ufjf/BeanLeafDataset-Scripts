from ast import Lambda
import cv2
from matplotlib import scale
import numpy as np
import glob
from bs4 import BeautifulSoup as bs
from PIL import Image, ImageOps
import os

error = 0

height1 = 4624
width1 = 3468
ratio1 = height1/width1

height2 = 512
width2 = 512

# Read images list
listIMGs = ''
# Open the file in read mode
with open('semmasc.txt', 'r') as file:
    # Read the contents of the file
    listIMGs = file.read()

if width2 <= height2:
    path = '../Bean leaf dataset/'
    for i in range(128, 129):
        leafNumber = i
        print( str(leafNumber).zfill(3))

        # Get all marker xml files for the current leaf
        files = glob.glob(path + str(leafNumber).zfill(3) + '/marker/*.xml')

        for fname in files:
            filename = (fname.split('marker\\')[1]).replace('.xml', '.jpg')
            if filename in listIMGs:
                print('>>>', filename)

                # Read the marker XML file
                content = []
                with open(fname, "r") as file:
                    # Read each line in the file, readlines() returns a list of lines
                    content = file.readlines()
                    # Combine the lines in the list into a string
                    content = "".join(content)
                    bs_content = bs(content, "lxml")
                marker = bs_content.find("corners")
                result_marker = list(marker.strings)
                try:
                    while True:
                        result_marker.remove('\n')
                except ValueError:
                    pass
                marker_x = list(np.float_(result_marker[::2]))
                marker_y = list(np.float_(result_marker[1::2]))


                # Read the leaf XML file
                content = []
                with open(fname.replace('marker', 'leaf'), "r") as file:
                    # Read each line in the file, readlines() returns a list of lines
                    content = file.readlines()
                    # Combine the lines in the list into a string
                    content = "".join(content)
                    bs_content = bs(content, "lxml")

                leaf = bs_content.find("points")
                result_leaf = list(leaf.strings)
                try:
                    while True:
                        result_leaf.remove('\n')
                except ValueError:
                    pass
                leaf_x = list(np.float_(result_leaf[::2]))
                leaf_y = list(np.float_(result_leaf[1::2]))

                # Calculate the bounding box
                joined_x = marker_x + leaf_x
                joined_y = marker_y + leaf_y
                xmin = min(joined_x) * height1 #left
                ymin = min(joined_y) * height1 #top
                xmax = max(joined_x) * height1 #right
                ymax = max(joined_y) * height1 #bottom
                width = xmax - xmin
                height = ymax - ymin
                ratio = height/width
                center_x = (xmin + xmax)/2
                center_y = (ymin + ymax)/2
                start_y = int(center_y - (width1/2))

                if(start_y <= 0):
                    start_y = 0
                    end_y = width1
                else:
                    start_y = int(ymin - abs(start_y - ymin))
                    end_y = int(start_y + width1)

                if( start_y < ymin and end_y > ymax):
                    image_path = fname.replace('/marker', '')
                    image_path = image_path.replace('.xml', '.jpg')
                    new_imagePath = image_path.replace('/' + str(leafNumber).zfill(3), '/train/imgs')

                    # create dir
                    if not os.path.exists(new_imagePath.split('\\')[0]):
                        os.makedirs(new_imagePath.split('\\')[0])
                
                    image = cv2.imread(image_path)
                    cropped_image = image[start_y:end_y,0:width1]

                    with open("../new_origin_images.txt", "a") as arquivo:
                        arquivo.writelines(
                        "'" + filename + "': " + str(start_y)  + ',\n'
                        )

                    resized_image = cv2.resize(cropped_image, (width2, width2), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(new_imagePath, resized_image) 
                    
                    joined_y = list(map(lambda x: (x * height1) - start_y, joined_y))
                    joined_x = list(map(lambda x: (x * height1), joined_x))
                    joined_y_normalize = list(map(lambda x: (x / width1), joined_y))
                    joined_x_normalize = list(map(lambda x: (x / width1), joined_x))

                    pts1 = np.float32((joined_x, joined_y)).T 
                    pts2 = np.float32((np.array(joined_x_normalize) * width2 , np.array(joined_y_normalize) * height2)).T 

                    # Create leaf and marker masks by drawing white polygons on black images
                    new_imagePath = new_imagePath.replace('.jpg', '.png')  
                    new_imagePath = new_imagePath.replace('imgs', 'mask')        
        
                    #create dir
                    if not os.path.exists(new_imagePath.split('\\')[0]):
                        os.makedirs(new_imagePath.split('\\')[0])

                    mask = np.zeros(resized_image.shape, np.uint8)
                    cv2.fillPoly(mask, np.int32([pts2[:4]]), (0, 255, 0)) #marker
                    cv2.fillPoly(mask, np.int32([pts2[4:]]),  (0, 0, 255)) #leaf
                    cv2.imwrite(new_imagePath, mask)

                    color = (255, 255, 255)  
                    marker_mask = np.zeros(resized_image.shape, np.uint8)
                    cv2.fillPoly(marker_mask, np.int32([pts2[:4]]), color)
                    cv2.imwrite(new_imagePath.replace('mask', 'marker'), marker_mask)

                    leaf_mask = np.zeros(resized_image.shape, np.uint8)
                    cv2.fillPoly(leaf_mask, np.int32([pts2[4:]]), color)
                    cv2.imwrite(new_imagePath.replace('mask', 'leaf'), leaf_mask)

                else:
                    error += 1
                    pass
else:
    pass

print('N. errors: ', error)





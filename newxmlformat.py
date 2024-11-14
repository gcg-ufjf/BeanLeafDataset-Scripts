
import numpy as np
from numpy.linalg import inv
import cv2
import cv2.aruco as aruco
import glob
import xml.etree.ElementTree as ET
from new_origin_images_dict import *
import os
from dimensions_dict import *
import re

# Load camera matrices
cameraMatrix = np.load("Original camera matrix.npy")
dist = np.load("Distortion coefficients.npy")
newCameraMatrix = np.load("Optimal camera matrix.npy")

markerLength = 50

# Function to get corners in camera world
#https://stackoverflow.com/questions/46363618/aruco-markers-with-opencv-get-the-3d-corner-coordinates
def getCornersInCameraWorld(markerLength, rvec, tvec):
    half_side = markerLength / 2
    rot_mat, _ = cv2.Rodrigues(rvec)
    rot_mat_t = np.transpose(rot_mat)
    # E-0
    tmp = rot_mat_t[:, 0]
    camWorldE = np.array([tmp[0] * half_side,
                          tmp[1] * half_side,
                          tmp[2] * half_side])
    # F-0
    tmp = rot_mat_t[:, 1]
    camWorldF = np.array([tmp[0] * half_side,
                          tmp[1] * half_side,
                          tmp[2] * half_side])
    tvec_3f = np.array([tvec[0][0][0],
                        tvec[0][0][1],
                        tvec[0][0][2]])
    nCamWorldE = np.multiply(camWorldE, -1)
    nCamWorldF = np.multiply(camWorldF, -1)
    ret = np.array([tvec_3f, tvec_3f, tvec_3f, tvec_3f])
    ret[0] += np.add(nCamWorldE, camWorldF)
    ret[1] += np.add(camWorldE, camWorldF)
    ret[2] += np.add(camWorldE, nCamWorldF)
    ret[3] += np.add(nCamWorldE, nCamWorldF)
    return ret

# Read images list
listIMGs = ''
# Open the file in read mode
with open('list1-2-3-0.txt', 'r') as file:
    # Read the contents of the file
    listIMGs = file.read()

# Path to dataset images
path = '../Bean leaf dataset/'
for i in range(1, 613):
    leafNumber = str(i).zfill(3)
    print( leafNumber)

    # Get all image files for the current leaf
    files = glob.glob(path + leafNumber + '/*.jpg')
    
    # Loop through each leaf files
    for image_path in files:
        filename = (image_path.split('/' + leafNumber +'\\')[1]).replace('.jpg', '')
        if filename in listIMGs:
            print('>>> ', filename)

            # Parse the XML file corresponding to the image
            marker_xml_path = image_path.replace('/' + leafNumber, '/' + leafNumber + '/marker').replace('.jpg', '.xml')
            tree = ET.parse(marker_xml_path)
            root_marker = tree.getroot()

            # Extract filename from the XML
            filename_tag = root_marker.find('.//filename')
            filename_tag = filename_tag.text

            # Extract leaf points from the XML
            leaf_tag = root_marker.find('.//leaf')
            leaf_tag = leaf_tag.text

            # Extract marker corner points from the XML
            corners_tag = root_marker.find('.//corners')
            corners_x = []
            corners_y = []
            for i in range(1, 5):
                corners_x.append( np.double(corners_tag.find(f'x{i}').text) )
                corners_y.append( np.double(corners_tag.find(f'y{i}').text) )


            # Define new corners points
            new_corners_x = []
            new_corners_x.append(corners_x[1])
            new_corners_x.append(corners_x[2])
            new_corners_x.append(corners_x[3])
            new_corners_x.append(corners_x[0])

            new_corners_y = []
            new_corners_y.append(corners_y[1])
            new_corners_y.append(corners_y[2])
            new_corners_y.append(corners_y[3])
            new_corners_y.append(corners_y[0])

            # Scale the corner coordinates to match the original image dimensions
            corners = np.dstack((new_corners_x, new_corners_y)) * 4624

            # Estimate the pose of the object using solvePNP
            # Calculate the 3D coordinates of the Aruco marker
            object_points = np.array([[-markerLength/2,  markerLength/2, 0], 
                                    [ markerLength/2,  markerLength/2, 0], 
                                    [ markerLength/2, -markerLength/2, 0], 
                                    [-markerLength/2, -markerLength/2, 0]], dtype=np.float32)
            
            success, rvec, tvec = cv2.solvePnP(object_points, 
                                            corners, 
                                            cameraMatrix, 
                                            dist, 
                                            flags=cv2.SOLVEPNP_IPPE_SQUARE)
            
            rotation_mat, _ = cv2.Rodrigues(rvec)
            tvec = np.array([[np.squeeze(tvec)]], dtype=np.double)
            rvec = np.array([[np.squeeze(rvec)]], dtype=np.double)
            # Get corners in camera world coordinates
            ret = getCornersInCameraWorld(markerLength, rvec, tvec)
            # Create the homogeneous transformation matrix
            homogeneous_mat = np.column_stack((rotation_mat, tvec.reshape(1,3).T))
            extrinsic_mat = np.concatenate((homogeneous_mat, [(0, 0, 0, 1)]), axis=0)
            # Compute the perspective matrix
            perspective_mat = np.dot(cameraMatrix, homogeneous_mat)

            leaf_xml_path = image_path.replace('/' + leafNumber, '/' + leafNumber + '/leaf').replace('.jpg', '.xml')
            tree = ET.parse(leaf_xml_path)
            root_leaf = tree.getroot()

            points_tag = root_leaf.find('.//points')
            points_x = []
            points_y = []

            # Find all elements with names starting with 'x'
            x_elements = [elem for elem in points_tag.iter() if elem.tag.startswith('x')]
            n = len(x_elements) + 1
            for i in range(1, n):
                points_x.append( np.double(points_tag.find(f'x{i}').text) )
                points_y.append( np.double(points_tag.find(f'y{i}').text) )

            area = str(area_dict[str(leafNumber)])
            width = str(width_dict[str(leafNumber)])
            length = str(length_dict[str(leafNumber)])
            perimeter = str(perimeter_dict[str(leafNumber)])

            folder_annotation = path + leafNumber + '/annotation/'
            print(folder_annotation)

            # Check if the folder already exists
            if not os.path.exists(folder_annotation):
                # Create the folder
                os.mkdir(folder_annotation)

            headerXML = '''<annotation>\n  <filename> ''' + str(filename_tag) + ''' </filename>'''
            headerXML += '''\n  <image-size>\n    <width> 3468 </width>\n    <height> 4624 </height>\n    <depth> 3 </depth>\n  </image-size>'''
            headerXML += '''\n  <objects>\n    <leaf-number> ''' + leafNumber + ''' </leaf-number>\n    <marker>'''
            headerXML += '''\n      <area> 25 </area>    \n      <normalized-original-corners>\n'''

            elements_marker = ""
            for i in range(0, 4):
                elements_marker = elements_marker + '        <x%s>%s</x%s>\n          <y%s>%s</y%s>\n'% (i+1, corners_x[i], i+1 , i+1 , corners_y[i], i+1)
            elements_marker += '''      </normalized-original-corners>'''
            
            elements_marker += '''\n      <normalized-rotated-corners>\n'''
            for i in range(0, 4):
                elements_marker = elements_marker + '        <x%s>%s</x%s>\n          <y%s>%s</y%s>\n'% (i+1, new_corners_x[i], i+1 , i+1 , new_corners_y[i], i+1)
            elements_marker += '''      </normalized-rotetad-corners>\n    </marker>'''

            elements_marker += '''\n    <leaf>\n      <dimensions>'''
            elements_marker += '''\n        <area> ''' + area + ''' </area>'''
            elements_marker += '''\n        <width> ''' + length + ''' </width>'''
            elements_marker += '''\n        <length> ''' + width + ''' </length>'''
            elements_marker += '''\n        <perimeter> ''' + perimeter + ''' </perimeter>'''
            elements_marker += '''\n      </dimensions>\n      <normalized-polygon>\n'''

            elements_leaf = ""
            for i in range(0, n-1):
                elements_leaf = elements_leaf + '        <x%s>%s</x%s>\n          <y%s>%s</y%s>\n'% (i+1, points_x[i], i+1 , i+1 , points_y[i], i+1)
            elements_leaf += '''      </normalized-polygon>\n    </leaf>  \n  </objects>\n  '''

            calibration_tags = '''<calibration>\n    <camera-matrix>\n      <rows> 3 </rows>\n      <cols> 3 </cols>\n      <data>'''
            calibration_tags += '''\n        3507.61282154  0.  1729.16383347
        0.  3491.16758467  2304.25914165
        0.  0.  1.'''

            calibration_tags += '''\n      </data>\n    </camera-matrix>\n    <distortion-coefficients>'''
            for linha in dist:
                for valor in linha:
                    calibration_tags += '''\n      <item> %s </item>'''% (valor)

            calibration_tags += '''\n    </distortion-coefficients>\n    <tvec>\n      '''
            calibration_tags += ' '.join(str(num) for num in tvec[0, -1])
            calibration_tags += '''\n    </tvec>\n    <rvec>\n      '''
            calibration_tags += ' '.join(str(num) for num in rvec[0, -1])
            calibration_tags += '''\n    </rvec>\n    <rotation-matrix>\n      <rows> 3 </rows>\n      <cols> 3 </cols>\n      <data>'''
            
            data = str(rotation_mat).replace('[', '').replace(']', '')
            data = data.replace('\n', '\n      ')
            calibration_tags += '''\n       ''' + data
            calibration_tags += '''\n      </data>\n    </rotation-matrix>\n    <extrinsic-homogeneous-matrix>\n      <rows> 4 </rows>\n      <cols> 4 </cols>\n      <data>'''

            data = str(homogeneous_mat).replace('[', '').replace(']', '')
            data = data.replace('\n', '\n      ')
            calibration_tags += '''\n       ''' + data
            calibration_tags += '''\n        0  0  0  1'''
            calibration_tags += '''\n      </data>\n    </extrinsic-homogeneous-matrix>\n    <model-projection-matrix>\n      <rows> 3 </rows>\n      <cols> 4 </cols>\n      <data>'''
            
            data = str(perspective_mat).replace('[', '').replace(']', '')
            data = data.replace('\n', '\n      ')
            calibration_tags += '''\n       ''' + data
            calibration_tags +=  '''\n      </data>\n    </model-projection-matrix>\n  </calibration>\n</annotation>'''
            
            # Combine XML content with calibration data, write it to the original XML file, and close the file
            XML = (headerXML+elements_marker+elements_leaf+calibration_tags)
            outFile = open(folder_annotation + filename + '.xml',"w")
            #outFile = open(fname.replace(".xml", '_modificado.xml'),"w")
            outFile.write(XML)
            outFile.close()

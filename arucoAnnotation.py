import numpy as np
from ast import Lambda
import cv2
import cv2.aruco as aruco
from matplotlib import scale
import glob
from bs4 import BeautifulSoup as bs
#https://linuxhint.com/parse_xml_python_beautifulsoup/
import xml.etree.ElementTree as ET
import re

np.set_printoptions(suppress=True)

# Load camera matrices
cameraMatrix = np.load("Original camera matrix.npy")
dist = np.load("Distortion coefficients.npy")
newCameraMatrix = np.load("Optimal camera matrix.npy")

# Define custom ArUco dictionary
aruco_dict = aruco.custom_dictionary(0, 3, 1)
aruco_dict.bytesList = np.empty(shape=(1, 2, 4), dtype=np.uint8)

# Add new marker to the dictionary
mybits = np.array([[1, 0, 1], [1, 0, 0], [1, 1, 1]], dtype=np.uint8)
aruco_dict.bytesList[0] = aruco.Dictionary_getByteListFromBits(mybits)

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

# Path to dataset images
path = '../../Bean leaf dataset/'
for i in range(1, 151):
    leafNumber = i
    print( str(leafNumber).zfill(3))

    # Glob for marker XML files
    files = glob.glob(path + str(leafNumber).zfill(3) + '/marker/*.xml')

    for fname in files:
        newFname = (fname.replace("marker\\", "")).replace(".xml", ".jpg")
        #print(fname, "\n")
        frame = cv2.imread(newFname)

        # Read XML content and extract marker coordinates
        content = []
        with open(fname, "r") as file:
            content = file.readlines()
            content = "".join(content)
            bs_content = bs(content, "lxml")
        marker = bs_content.find("corners")
        result_marker = list(marker.strings)
        try:
            while True:
                result_marker.remove('\n')
        except ValueError:
            pass

        marker_x = np.array(np.float_(result_marker[::2]), dtype=np.float32)
        marker_y = np.array(np.float_(result_marker[1::2]), dtype=np.float32)
        corners = np.dstack((marker_x,marker_y)) * 4624

        # Estimate marker pose
        rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(
            corners,
            markerLength,
            cameraMatrix,
            dist
        )
        (rvec - tvec).any()  # get rid of that nasty numpy value array error
        aruco.drawAxis(frame, cameraMatrix, dist, rvec, tvec, markerLength)  # Draw axis

        # Compute camera world coordinates and matrices
        rotation_mat, _ = cv2.Rodrigues(rvec)
        ret = getCornersInCameraWorld(markerLength, rvec, tvec)
        homogeneous_mat = np.column_stack((rotation_mat, tvec.reshape(1,3).T))
        extrinsic_mat = np.concatenate((homogeneous_mat, [(0, 0, 0, 1)]), axis=0)
        perspective_mat = np.dot(cameraMatrix, homogeneous_mat)
        
        # Modify XML content with calibration data
        tree = ET.parse(fname)
        root = tree.getroot()
        for tag in root.findall('calibration'):
            root.remove(tag)
        xml_string = ET.tostring(root, encoding='unicode')
        xml_string = xml_string.replace('.png', '.jpg')
        xml_string = xml_string.replace('</annotation>', '')
        new_tags = '''<calibration>\n    <camera-matrix>\n      <rows>3</rows>\n      <cols>3</cols>\n      <data>'''

        # Append camera matrix data to XML
        new_tags += '''\n        3507.61282154  0.  1729.16383347
        0.  3491.16758467  2304.25914165
        0.  0.  1.'''
        new_tags += '''\n      </data>\n    </camera-matrix>\n    <distortion-coefficients>'''
        for line in dist:
            for value in line:
                new_tags += '''\n      <item>%s</item>'''% (value)

        new_tags += '''\n    </distortion-coefficients>\n    <tvec>\n      '''
        new_tags += ' '.join(str(num) for num in tvec[0, -1])
        new_tags += '''\n    </tvec>\n    <rvec>\n      '''
        new_tags += ' '.join(str(num) for num in rvec[0, -1])
        new_tags += '''\n    </rvec>\n    <rotation-matrix>\n      <rows>3</rows>\n      <cols>3</cols>\n      <data>'''
        
        # Append rotation matrix data to XML

        if ('-' in str(rotation_mat[0])):
            string = str(rotation_mat[0]).replace('[', '').replace(']', '')
            new_tags += '''\n       %s'''% (string)
        else:
            string = str(rotation_mat[0]).replace('[', '').replace(']', '')
            new_tags += '''\n        %s'''% (string)
        string = str(rotation_mat[1]).replace('[', '').replace(']', '')
        new_tags += '''\n       %s'''% (string)
        string = str(rotation_mat[2]).replace('[', '').replace(']', '')
        new_tags += '''\n       %s'''% (string)
        new_tags += '''\n      </data>\n    </rotation-matrix>\n    <extrinsic-homogeneous-matrix>\n      <rows>4</rows>\n      <cols>4</cols>\n      <data>'''

        # Append extrinsic matrix data to XML
        homogeneous_mat = re.sub('\n ', '\n', str(homogeneous_mat).replace('[','').replace(']',''))
        homogeneous_mat = re.sub(' +', '  ', str(homogeneous_mat))
        new_tags += '''\n      ''' + homogeneous_mat.split("\n")[0]
        new_tags += '''\n      ''' + homogeneous_mat.split("\n")[1]
        new_tags += '''\n      ''' + homogeneous_mat.split("\n")[2]

        new_tags += '''\n        0  0  0  1'''
        new_tags += '''\n      </data>\n    </extrinsic-homogeneous-matrix>\n    <model-projection-matrix>\n      <rows>3</rows>\n      <cols>4</cols>\n      <data>'''
        
        # Append model projection matrix data to XML
        for line in perspective_mat:
            line = re.sub(' +', '  ', str(line))
            new_tags += '''\n      ''' + str(line).replace('[','').replace(']','')
        footerXML =  '''\n      </data>\n    </model-projection-matrix>\n  </calibration>\n</annotation>'''

        # Combine XML content with calibration data, write it to the original XML file, and close the file
        XML = (xml_string + new_tags + footerXML)
        outFile = open(fname,"w")
        outFile.write(XML)
        outFile.close()

import numpy as np
import cv2
import cv2.aruco as aruco
import glob
import xml.etree.cElementTree as ET
import os

# Load camera matrices
cameraMatrix = np.load("Original camera matrix.npy")
dist = np.load("Distortion coefficients.npy")
newCameraMatrix = np.load("Optimal camera matrix.npy")

markerLength = 50

# Define ArUco detector parameters
arucoParams = aruco.DetectorParameters_create()

# Set auxiliary parameters for ArUco marker detection
auxParam = 0 

# Adjust parameters based on the chosen auxParam set
if(auxParam != 0):
    arucoParams.minMarkerPerimeterRate = 0.09 #default 0.03
    arucoParams.polygonalApproxAccuracyRate = 0.1 #default 0.03 0.1
    arucoParams.maxErroneousBitsInBorderRate = 0.30 #default 0.35

if (auxParam == 1):
    arucoParams.maxMarkerPerimeterRate = 2.0 #default 4.0
    arucoParams.adaptiveThreshWinSizeMax = 393  #default 23 
    arucoParams.adaptiveThreshWinSizeMin = 83  #default 3 83

elif (auxParam == 2):
    arucoParams.maxMarkerPerimeterRate = 2.0 #default 4.0
    arucoParams.adaptiveThreshWinSizeMin = 5
    arucoParams.cornerRefinementMethod = 2
    
elif (auxParam == 3):
    arucoParams.maxMarkerPerimeterRate = 1.85 #default 4.0
    arucoParams.adaptiveThreshWinSizeMax = 755 
    arucoParams.adaptiveThreshWinSizeMin = 55

    #arucoParams.adaptiveThreshWinSizeMax = 95  
    arucoParams.adaptiveThreshConstant = 0.5
    #arucoParams.cornerRefinementMethod = 2

# Define custom ArUco dictionary 
aruco_dict = aruco.custom_dictionary(0, 3, 1)
aruco_dict.bytesList = np.empty(shape = (1, 2, 4), dtype = np.uint8)

# Add new marker to the dictionary
mybits = np.array([[1,0,1],[1,0,0],[1,1,1]], dtype = np.uint8)
aruco_dict.bytesList[0] = aruco.Dictionary_getByteListFromBits(mybits)

# Save marker image
#for i in range(len(aruco_dict.bytesList)):
#    cv2.imwrite("custom_aruco_" + str(i) + ".png", aruco.drawMarker(aruco_dict, i, 128))

# Define a function to calculate corner coordinates in camera world
#https://stackoverflow.com/questions/46363618/aruco-markers-with-opencv-get-the-3d-corner-coordinates
def getCornersInCameraWorld(markerLength, rvec, tvec):
    ##Pinhole camera model
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
path = '../Bean leaf dataset/'

# Loop over a range of leaf numbers (adjust as needed)
for i in range(35, 36):
    leafNumber = str(i).zfill(3)
    print(leafNumber)

    # Get the names of all images in the folder
    images = glob.glob(path + leafNumber + '/*.jpg')

    # Loop over each image
    for fname in images:
        print(fname)
        frame = cv2.imread(fname)

        # Detect ArUco markers in the image
        if(auxParam == 0): corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict)
        else: corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=arucoParams)

        if np.all(ids is not None):  # If there are markers found by detector
            
            # Create a black mask image with the same shape as the frame
            mask = mask = np.zeros(frame.shape, np.uint8)

            # Copy and reshape the corner points
            copyCorners = np.array(corners[0][0], dtype=np.int32)
            copyCorners = copyCorners.reshape(-1,1,2)

            # Fill the polygon defined by the corner points with the specified color in the mask
            color = (255, 255, 255)
            mask = cv2.fillPoly(mask, [copyCorners], color)
        
            # Find contours in the mask
            imgray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(imgray, 127, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            # Draw contours on the original frame
            cv2.drawContours(frame, contours, -1, (255, 100, 0), 5) 

            # Save the frame with contours drawn to the new path
            if not os.path.exists(str(fname.split("\\")[0]) + "\marker"):
                os.makedirs(str(fname.split("\\")[0]) + "\marker")
            newPath = fname.replace(leafNumber + "\\", leafNumber + "\marker\\")
            cv2.imwrite(newPath, frame)

            # Normalize corner points
            h = 4624
            normalize = copyCorners/h

            headerXML = '''<annotation>\n  <filename>''' + str(fname.split("\\")[1]) + '''</filename>\n  <object>\n    <leaf> ''' + leafNumber + ''' </leaf>\n    <corners>\n'''
            i = 1
            elements = ""
            for element in normalize:
                elements = elements+ '      <x%s>%s</x%s>\n        <y%s>%s</y%s>\n'% (i, element[0][0], i , i , element[0][1], i)
                i += 1
            footerXML = '''    </corners>\n  </object>\n</annotation>'''

            # Combine XML content with calibration data, write it to the original XML file, and close the file
            XML = (headerXML+elements+footerXML)
            outFile = open(newPath.replace("jpg","xml"),"w")
            outFile.write(XML)
            outFile.close()
        else:
            print("Error")

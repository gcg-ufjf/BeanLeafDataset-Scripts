import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob 

print(cv.__version__)

# Termination criteria for obtaining subpixel contours
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Define points in the object as (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Vectors to establish correspondences in each image.
objpoints = [] # 3D points of the object
imgpoints = [] # 2D points in the image plane

images = glob.glob('Calibration/*.jpg')
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the corners of the chessboard
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        # Add object points
        objpoints.append(objp)
        # Refine corners
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        # Add image points
        imgpoints.append(corners)
        # Draw and show the corners
        cv.drawChessboardCorners(img, (7,6), corners2, ret)

        plt.imshow(img)
        plt.title(fname)
        plt.xticks([]), plt.yticks([])
        plt.show()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

for fname in images:
    img = cv.imread(fname)
    dst = cv.undistort(img, mtx, dist)
    plt.imshow(img)
    plt.title(fname + " - original")
    plt.xticks([]), plt.yticks([])
    plt.show()
    plt.imshow(dst)
    plt.title(fname + " - undistorted")
    plt.xticks([]), plt.yticks([])
    plt.show()
import numpy as np
import cv2 as cv
import glob

chessboardSize = (9,6)
frameSize = (3468,4624)

# Termination criteria for obtaining subpixel contours
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Define points in the object as (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 15
objp = objp * size_of_chessboard_squares_mm

# Vectors to establish correspondences in each image.
objpoints = [] # 3D points of the object
imgpoints = [] # 2D points in the image plane

images = glob.glob('Calibration/*.jpg')
for image in images:
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        resize frame to show even on smaller screens
        show = cv.resize(img, (0,0), fx = 0.15, fy = 0.15)
        cv.imshow('img', show)
        cv.waitKey(1000)

        cv.imwrite(image.replace("Calibration", "a"), img)

cv.destroyAllWindows()

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
h = 4624
w = 3468
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

print("Camera Calibrated:\n", ret)
print("\nCamera Matrix:\n", cameraMatrix)
print("\nDistortion Parameters:\n", dist)
print("\nRotation Vectors:\n", rvecs)
print("\nTranslation Vectors:\n", tvecs)

np.save("Original camera matrix", cameraMatrix)
np.save("Distortion coefficients", dist)
np.save("Optimal camera matrix", newCameraMatrix)

# Reprojection Error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )

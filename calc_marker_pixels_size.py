import numpy as np
from numpy.linalg import inv
import cv2
import cv2.aruco as aruco
import glob
import xml.etree.ElementTree as ET
from new_origin_images_dict import * #generate by script 'resizeimages512x512.py'

# Load camera matrices
cameraMatrix = np.load("Original camera matrix.npy")
dist = np.load("Distortion coefficients.npy")
newCameraMatrix = np.load("Optimal camera matrix.npy")

Op = (0,0,0) # Observation point

# Define custom ArUco dictionary
aruco_dict = aruco.custom_dictionary(0, 3, 1)
aruco_dict.bytesList = np.empty(shape=(1, 2, 4), dtype=np.uint8)

# Add new marker to the dictionary
mybits = np.array([[1, 0, 1], [1, 0, 0], [1, 1, 1]], dtype=np.uint8)
aruco_dict.bytesList[0] = aruco.Dictionary_getByteListFromBits(mybits)

markerLength = 50
height = 4624
width = 3468

def get_pixel_corners(pixel):
    x, y = pixel
    return np.array([(x-0.5, y-0.5, 1.0),   #top_left
                 (x+0.5, y-0.5, 1.0),   #top_right   
                 (x+0.5, y+0.5, 1.0),   #bottom_right
                 (x-0.5, y+0.5, 1.0)], dtype=np.double)  #bottom_left

#https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html
#https://stackoverflow.com/questions/28011873/find-world-space-coordinate-for-pixel-in-opencv
def p2d_to_camera_coordinate(p2d, camera_matrix):
    p3d = np.dot(inv(camera_matrix), p2d)
    return p3d

def plane_equation(n, p):
    a, b, c = n
    #d = -1 * (a * p[0] + b * p[1] + c * p[2])
    d = -p.dot(n)
    return a, b, c, d

def line_plane_intersection(n, p, p1, p2): 
    a, b, c, d = plane_equation(n, p)
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    dx, dy, dz = x2-x1, y2-y1, z2-z1
    t = -(a*x1 + b*y1 + c*z1 + d) / (a*dx + b*dy + c*dz)
    x, y, z = x1 + t*dx, y1 + t*dy, z1 + t*dz
    return x, y, z

def triangle_area(A, B, C):
    # Calculate vectors AB and AC
    AB = [B[0]-A[0], B[1]-A[1], B[2]-A[2]]
    AC = [C[0]-A[0], C[1]-A[1], C[2]-A[2]]
    # Calculate cross product of AB and AC
    cross_product = [AB[1]*AC[2] - AB[2]*AC[1], 
                 AB[2]*AC[0] - AB[0]*AC[2], 
                 AB[0]*AC[1] - AB[1]*AC[0]]
    # Calculate area of triangle as half of the magnitude of the cross product
    area = 0.5 * (cross_product[0]**2 + cross_product[1]**2 + cross_product[2]**2)**0.5
    return area

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

# Read images list without '.jpg'
listIMGs = ''
# Open the file in read mode
with open('listIMGs.txt', 'r') as file:
    # Read the contents of the file
    listIMGs = file.read()

# Path to dataset images
path = '../Bean leaf dataset/'
for i in range(1, 129):
    leafNumber = str(i).zfill(3)
    print( leafNumber)

    # Get all image files for the current leaf
    files = glob.glob(path + leafNumber + '/*.jpg')
    
    # Loop through each leaf files
    for image_path in files:
        filename = (image_path.split('/' + leafNumber +'\\')[1]).replace('.jpg', '')
        if filename in listIMGs:
            try:
                # Get the new origin for the current image from the dictionary
                new_origin = imgs_dict[filename]
                print('>>> ', filename)

                # Parse the XML file corresponding to the image
                xml_path = image_path.replace('/' + leafNumber, '/' + leafNumber + '/marker').replace('.jpg', '.xml')
                tree = ET.parse(xml_path)
                root = tree.getroot()

                # Extract marker corner points from the XML
                corners_tag = root.find('.//corners')
                corners = []
                for i in range(1, 5):
                    x = (height * float(corners_tag.find(f'x{i}').text))
                    y = (height * float(corners_tag.find(f'y{i}').text)) 
                    z = 1.0
                    corners.append([x,y])

                corners = np.asarray([corners])
                #print(corners)

                # Change order of corner points
                corners = corners[0]
                corners = np.asarray([[
                            [corners[3][0], corners[3][1]],
                            [corners[2][0], corners[2][1]],
                            [corners[1][0], corners[1][1]],
                            [corners[0][0], corners[0][1]],
                            ]])

                # Possible: original(0-1-2-3), 0-3-2-1, 1-0-3-2, 1-2-3-0, 2-1-0-3, 2-3-0-1, 3-0-1-2, 3-2-1-0            
                
                # Estimate the pose of the object using solvePNP
                # Calculate the 3D coordinates of the Aruco marker
                object_points = np.array([[-markerLength/2,  markerLength/2, 0], 
                                    [ markerLength/2,  markerLength/2, 0], 
                                    [ markerLength/2, -markerLength/2, 0], 
                                    [-markerLength/2, -markerLength/2, 0]], dtype=np.float32)
                
                #image_points = corners[ids == marker_id][0]
                success, rvec_solve, tvec_solve = cv2.solvePnP(object_points, 
                                        corners, 
                                        cameraMatrix, 
                                        dist, 
                                        flags=cv2.SOLVEPNP_IPPE_SQUARE)
                #print('Sucess: ', success)
                rotation_mat_solve, _ = cv2.Rodrigues(rvec_solve)
                
                # Read original image
                original_image = cv2.imread(image_path)

                # Read mask and find indices where mask value is 255 (white)
                mask_path = '../Bean leaf dataset/marker/' + filename + '.png'
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                indices = np.where(mask == 255)
                pixels = list(zip(indices[1], indices[0]))

                # Create a buffer array with the same shape as the mask, initialized with zeros and of data type double
                buffer = np.zeros(mask.shape, np.double)

                # Iterate through each pixel in the list of white pixels
                for pixel in pixels:

                    # Get the corner points corresponding to the current pixel
                    pixel_corners = get_pixel_corners(pixel)

                    # Convert points to the scale of the original image
                    r = width/512
                    M = np.float32([[r, 0, 0],
                                [0, r, new_origin], 
                                [0 ,0 , 1]])

                    new_pixel_corners = []
                    for i in range(4):
                        new_pixel_corners.append(np.dot(M, pixel_corners[i].T))
                    pixel_corners = np.array(new_pixel_corners, dtype=np.double)

                    # Initialize a list to store intersection points
                    intersections = []

                    # Iterate through each corner point of the pixel
                    for p2d in pixel_corners:
                        # Convert input point to numpy array with correct shape
                        p2d = np.array([[p2d[:-1]]], dtype=np.double)

                        # Convert undistorted 2D point to camera coordinate
                        undistorted_pt = cv2.undistortPoints(p2d, cameraMatrix, dist, None, cameraMatrix)
                        p3d = p2d_to_camera_coordinate(np.hstack([undistorted_pt[0][0], 1.0]), cameraMatrix)
                        
                        # Find the intersection point of the line passing through the camera origin and the 3D point with the plane
                        intersections.append(line_plane_intersection(rotation_mat_solve[2], np.squeeze(tvec_solve), Op, p3d))
                        
                    # Calculate the area of the triangles formed by intersection points
                    area_triangle1 = triangle_area(intersections[0], intersections[1], intersections[2])
                    area_triangle2 = triangle_area(intersections[0], intersections[2], intersections[3])

                    # Assign the average of the two triangle areas to the corresponding pixel in the buffer, scaled by 100.0
                    buffer[pixel[1], pixel[0]] = (area_triangle1 + area_triangle2)/100.0
                                    
                new_image_path = image_path.replace('/' + leafNumber, '/700')
                
                # Calculate the total area from the buffer
                area = np.sum(buffer)
                
                # Check if the absolute difference between the calculated area and 25 is less than 1
                if abs(25 - area) < 1: 
                    print('Total area: ', area)
                    buffer.tofile(new_image_path.replace('.jpg','.raw'))
                else:
                    print('Dif > 1, Area: ', area)
                    #buffer.tofile(new_image_path.replace('.jpg','_difmaior1.raw'))
            except:
                print('Error')
                pass        

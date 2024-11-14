import xml.etree.ElementTree as ET
import glob
import numpy as np
from numpy.linalg import inv
import cv2.aruco as aruco
import cv2
from new_origin_images_dict import *

# Load camera matrices
cameraMatrix = np.load("Original camera matrix.npy")
dist = np.load("Distortion coefficients.npy")
newCameraMatrix = np.load("Optimal camera matrix.npy")

Op = (0, 0, 0) # Observation point

# Define custom ArUco dictionary
#aruco_dict = aruco.custom_dictionary(0, 3, 1)
aruco_dict = aruco.Dictionary_create(1, 3)
aruco_dict.bytesList = np.empty(shape=(1, 2, 4), dtype=np.uint8)

# Add new marker to the dictionary
mybits = np.array([[1, 0, 1], [1, 0, 0], [1, 1, 1]], dtype=np.uint8)
aruco_dict.bytesList[0] = aruco.Dictionary_getByteListFromBits(mybits)

img = aruco.drawMarker(aruco_dict, 0, 250)
cv2.imwrite(f"marker{0}.png", img)

markerLength = 50
height = 4624
width = 3468

def pixel_corners(x, y):
    return np.array([[x-0.5, y-0.5, 1.0],   #top_left
                     [x+0.5, y-0.5, 1.0],   #top_right   
                     [x+0.5, y+0.5, 1.0],   #bottom_right
                     [x-0.5, y+0.5, 1.0]])  #bottom_left

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
    # create a meshgrid for x and y
    x = np.linspace(-300, 300, 100)
    y = np.linspace(-300, 300, 100)
    X, Y = np.meshgrid(x, y)
    # calculate the corresponding z values for the plane
    Z = (-a*X - b*Y - d) / c
    # Define a direction vector for the line
    dir_vec = p2 - p1
    # Define the range of the line
    t = np.linspace(p1[0], 300, 100)
    # Compute the coordinates of the points along the line
    line_points = np.array([p1 + t_i * dir_vec for t_i in t])
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
with open('semmasc.txt', 'r') as file:
    # Read the contents of the file
    listIMGs = file.read()

# Path to dataset images
path = '../Bean leaf dataset/'
for i in range(1, 613):
    leafNumber = str(i).zfill(3)
    print( leafNumber)

    # Get all marker xml files for the current leaf
    files = glob.glob(path + leafNumber + '/marker/*.xml')

    #Loop through each xml files
    for xml_path in files:
        filename = (xml_path.split('marker\\')[1]).replace('.xml', '.jpg')
        if(filename in listIMGs):
            print('\n------------------------------------------------------------------------------------')
            print('>>>', filename)
            fname = path + leafNumber + '/' + filename
            newFname = path + '700/' + filename
            
            # Parse the XML file corresponding to the image
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Extract marker corner points from the XML
            corners_tag = root.find('.//corners')
            corners = []
            for i in range(1, 5):
                x = (height * float(corners_tag.find(f'x{i}').text))
                y = (height * float(corners_tag.find(f'y{i}').text)) 
                z = 1.0 # Set z-coordinate to 1.0 
                corners.append([x,y])
                p2d = np.array([x, y, z]) # Create a numpy array for 2D coordinates
                p3d = p2d_to_camera_coordinate(p2d, cameraMatrix) # Convert 2D to 3D coordinates

            corners_orig = np.asarray([corners])

            rotation = ''
            best_rotation = 'original'

            for ordem in range(7):
                original_image = cv2.imread(fname)

                '''cv2.circle(original_image, tuple(np.int32([corners[0][0], corners[0][1]])), radius=20, color=(0, 0, 255), thickness=-1)
                cv2.circle(original_image, tuple(np.int32([corners[1][0], corners[1][1]])), radius=20, color=(0, 255, 0), thickness=-1)
                cv2.circle(original_image, tuple(np.int32([corners[2][0], corners[2][1]])), radius=20, color=(255, 0, 0), thickness=-1)
                cv2.circle(original_image, tuple(np.int32([corners[3][0], corners[3][1]])), radius=20, color=(0, 255, 255), thickness=-1)
                    
                cv2.imwrite(fname.replace(".png", "_points_" + rotation + ".jpg"), original_image)
                '''
                
                # Initialize lists to store intersection points in different pose estimation
                intersections_sm_orig = []
                intersections_solve_orig = []
                intersections_sm_rot = []
                intersections_solve_rot = []

                if ordem == 0:
                    #Ordem 3-2-1-0
                    rotation = '3-2-1-0'
                    corners_rot = corners[0]
                    corners_rot = np.asarray([[
                                    [corners[3][0], corners[3][1]],
                                    [corners[2][0], corners[2][1]],
                                    [corners[1][0], corners[1][1]],
                                    [corners[0][0], corners[0][1]]
                                    ]])
                    
                elif ordem == 1:
                    #Ordem 3-0-1-2
                    rotation = '3-0-1-2'
                    corners_rot = corners[0]
                    corners_rot = np.asarray([[
                                    [corners[3][0], corners[3][1]],
                                    [corners[0][0], corners[0][1]],
                                    [corners[1][0], corners[1][1]],
                                    [corners[2][0], corners[2][1]]
                                    ]])
                    
                elif ordem == 2:
                    #Ordem 2-1-0-3
                    rotation = '2-1-0-3'
                    corners_rot = corners[0]
                    corners_rot = np.asarray([[
                                    [corners[2][0], corners[2][1]],
                                    [corners[1][0], corners[1][1]],
                                    [corners[0][0], corners[0][1]],
                                    [corners[3][0], corners[3][1]]
                                    ]])

                if ordem == 3:
                    #Ordem 600 2-3-0-1
                    rotation = '2-3-0-1'
                    corners_rot = corners[0]
                    corners_rot = np.asarray([[
                                    [corners[2][0], corners[2][1]],
                                    [corners[3][0], corners[3][1]],
                                    [corners[0][0], corners[0][1]],
                                    [corners[1][0], corners[1][1]]
                                    ]])
                    
                elif ordem == 4:
                    #Ordem 1-0-3-2
                    rotation = '1-0-3-2'
                    corners_rot = corners[0]
                    corners_rot = np.asarray([[
                                    [corners[1][0], corners[1][1]],
                                    [corners[0][0], corners[0][1]],
                                    [corners[3][0], corners[3][1]],
                                    [corners[2][0], corners[2][1]]
                                    ]])

                elif ordem == 5:
                    #Ordem 1-2-3-0
                    rotation = '1-2-3-0'
                    corners_rot = corners[0]
                    corners_rot = np.asarray([[
                                    [corners[1][0], corners[1][1]],
                                    [corners[2][0], corners[2][1]],
                                    [corners[3][0], corners[3][1]],
                                    [corners[0][0], corners[0][1]]
                                    ]])

                elif ordem == 6:
                    #Ordem 0-3-2-1
                    rotation = '0-3-2-1'
                    corners_rot = corners[0]
                    corners_rot = np.asarray([[
                                    [corners[0][0], corners[0][1]],
                                    [corners[3][0], corners[3][1]],
                                    [corners[2][0], corners[2][1]],
                                    [corners[1][0], corners[1][1]]
                                    ]])

                # Estimate pose of marker using aruco.estimatePoseSingleMarkers()
                # In original coordinates
                rvec_sm_orig, tvec_sm_orig, markerPoints = aruco.estimatePoseSingleMarkers(
                        [corners_orig],
                        markerLength,
                        cameraMatrix,
                        dist
                    )
                
                (rvec_sm_orig - tvec_sm_orig).any()  # get rid of that nasty numpy value array error

                # In rotated coordinates
                rvec_sm_rot, tvec_sm_rot, markerPoints = aruco.estimatePoseSingleMarkers(
                        [corners_rot],
                        markerLength,
                        cameraMatrix,
                        dist
                    )
                
                (rvec_sm_rot - tvec_sm_rot).any()  # get rid of that nasty numpy value array error

                # Convert rotation vectors to rotation matrices
                rotation_mat_sm_orig, _ = cv2.Rodrigues(rvec_sm_orig)
                rvec_sm_orig = rvec_sm_orig[0][0]
                tvec_sm_orig = tvec_sm_orig[0][0]
                
                rotation_mat_sm_rot, _ = cv2.Rodrigues(rvec_sm_rot)
                rvec_sm_rot = rvec_sm_rot[0][0]
                tvec_sm_rot = tvec_sm_rot[0][0]

                                
                
                # Estimate the pose of the object using solvePNP
                object_points = np.array([[-markerLength/2,  markerLength/2, 0], 
                                        [ markerLength/2,  markerLength/2, 0], 
                                        [ markerLength/2, -markerLength/2, 0], 
                                        [-markerLength/2, -markerLength/2, 0]], dtype=np.float32)
                
                # In original coordinates
                success, rvec_solve_orig, tvec_solve_orig = cv2.solvePnP(object_points, 
                                                corners_orig, 
                                                cameraMatrix, 
                                                dist, 
                                                flags=cv2.SOLVEPNP_IPPE_SQUARE)
                # Convert rotation vectors to rotation matrices
                rotation_mat_solve_orig, _ = cv2.Rodrigues(rvec_solve_orig)

                # In rotated coordinates
                success, rvec_solve_rot, tvec_solve_rot = cv2.solvePnP(object_points, 
                                                corners_rot, 
                                                cameraMatrix, 
                                                dist, 
                                                flags=cv2.SOLVEPNP_IPPE_SQUARE)
                # Convert rotation vectors to rotation matrices
                rotation_mat_solve_rot, _ = cv2.Rodrigues(rvec_solve_rot)
                

                # Iterate over each corner of the marker
                for i in range(1, 5):
                    x = (height * float(corners_tag.find(f'x{i}').text))
                    y = (height * float(corners_tag.find(f'y{i}').text)) 
                    z = 1.0

                    # Convert the 2D point to camera coordinate system
                    p2d = np.array([x, y, z])
                    p3d = p2d_to_camera_coordinate(p2d, cameraMatrix)

                    # Calculate intersection points with the ground plane for original and rotated markers
                    intersections_sm_orig.append(line_plane_intersection(rotation_mat_sm_orig[2], tvec_sm_orig, Op, p3d))
                    intersections_sm_rot.append(line_plane_intersection(rotation_mat_sm_rot[2], tvec_sm_rot, Op, p3d))
                
                    intersections_solve_orig.append(line_plane_intersection(rotation_mat_solve_orig[2], np.squeeze(tvec_solve_orig), Op, p3d))
                    intersections_solve_rot.append(line_plane_intersection(rotation_mat_solve_rot[2], np.squeeze(tvec_solve_rot), Op, p3d))
                    
                
                if ordem == 0:
                    intersections_sm_rot_copy = np.asarray([
                                    intersections_sm_rot[3],
                                    intersections_sm_rot[2],
                                    intersections_sm_rot[1],
                                    intersections_sm_rot[0]
                                    ])
                    intersections_solve_rot_copy = np.asarray([
                                    intersections_solve_rot[3],
                                    intersections_solve_rot[2],
                                    intersections_solve_rot[1],
                                    intersections_solve_rot[0]
                                    ])
                    
                elif ordem == 1:
                    intersections_sm_rot_copy = np.asarray([
                                    intersections_sm_rot[3],
                                    intersections_sm_rot[0],
                                    intersections_sm_rot[1],
                                    intersections_sm_rot[2]
                                    ])
                    intersections_solve_rot_copy = np.asarray([
                                    intersections_solve_rot[3],
                                    intersections_solve_rot[0],
                                    intersections_solve_rot[1],
                                    intersections_solve_rot[2]
                                    ])
                    
                elif ordem == 2:
                    intersections_sm_rot_copy = np.asarray([
                                    intersections_sm_rot[2],
                                    intersections_sm_rot[1],
                                    intersections_sm_rot[0],
                                    intersections_sm_rot[3]
                                    ])
                    intersections_solve_rot_copy = np.asarray([
                                    intersections_solve_rot[2],
                                    intersections_solve_rot[1],
                                    intersections_solve_rot[0],
                                    intersections_solve_rot[3]
                                    ])
                    
                if ordem == 3:
                    intersections_sm_rot_copy = np.asarray([
                                    intersections_sm_rot[2],
                                    intersections_sm_rot[3],
                                    intersections_sm_rot[0],
                                    intersections_sm_rot[1]
                                    ])
                    intersections_solve_rot_copy = np.asarray([
                                    intersections_solve_rot[2],
                                    intersections_solve_rot[3],
                                    intersections_solve_rot[0],
                                    intersections_solve_rot[1]
                                    ])
                    
                elif ordem == 4:
                    intersections_sm_rot_copy = np.asarray([
                                    intersections_sm_rot[1],
                                    intersections_sm_rot[0],
                                    intersections_sm_rot[3],
                                    intersections_sm_rot[2]
                                    ])
                    intersections_solve_rot_copy = np.asarray([
                                    intersections_solve_rot[1],
                                    intersections_solve_rot[0],
                                    intersections_solve_rot[3],
                                    intersections_solve_rot[2]
                                    ])

                elif ordem == 5:
                    intersections_sm_rot_copy = np.asarray([
                                    intersections_sm_rot[1],
                                    intersections_sm_rot[2],
                                    intersections_sm_rot[3],
                                    intersections_sm_rot[0]
                                    ])
                    intersections_solve_rot_copy = np.asarray([
                                    intersections_solve_rot[1],
                                    intersections_solve_rot[2],
                                    intersections_solve_rot[3],
                                    intersections_solve_rot[0]
                                    ])

                elif ordem == 6:
                    intersections_sm_rot_copy = np.asarray([
                                    intersections_sm_rot[0],
                                    intersections_sm_rot[3],
                                    intersections_sm_rot[2],
                                    intersections_sm_rot[1]
                                    ])
                    intersections_solve_rot_copy = np.asarray([
                                    intersections_solve_rot[0],
                                    intersections_solve_rot[3],
                                    intersections_solve_rot[2],
                                    intersections_solve_rot[1]
                                    ])
                    
                # Intersection Single Markers orig RED
                intersections_sm_orig2d = []
                for p in intersections_sm_orig:
                    aux = np.dot(cameraMatrix, p) / p[2]
                    aux_woZ = np.delete(aux, -1, axis=0)
                    intersections_sm_orig2d.append(aux_woZ)
                    #cv2.circle(original_image, tuple(np.int32(aux_woZ)), radius=20, color=(0, 0, 255), thickness=-1)

                intersections_sm_orig2d = np.asarray(intersections_sm_orig2d) 
                area_triangle1 = triangle_area(intersections_sm_orig[0], intersections_sm_orig[1], intersections_sm_orig[2])
                area_triangle2 = triangle_area(intersections_sm_orig[0], intersections_sm_orig[3], intersections_sm_orig[2])
                area_sm_orig = (area_triangle1 + area_triangle2)/100

                # RET with SingleMarkers orig GREEN
                ret = getCornersInCameraWorld(markerLength, rvec_sm_orig, [[tvec_sm_orig]])
                area_triangle1 = triangle_area(ret[0], ret[1], ret[2])
                area_triangle2 = triangle_area(ret[0], ret[2], ret[3])
                area_ret_sm_orig = (area_triangle1 + area_triangle2)/100
                #print('Area IDEAL(Single Markers orig): ', area_ret_sm_orig)

                aux = np.dot(cameraMatrix, tvec_sm_orig) / tvec_sm_orig[2]
                aux_woZ = np.delete(aux, -1, axis=0)
                #cv2.circle(original_image, tuple(np.int32(aux_woZ)), radius=20, color=(255, 0, 0), thickness=-1)

                for p in ret:
                    aux = np.dot(cameraMatrix, p) / p[2]
                    aux_woZ = np.delete(aux, -1, axis=0)
                    #cv2.circle(original_image, tuple(np.int32(aux_woZ)), radius=20, color=(0, 0, 255), thickness=-1)
                
                intersections_sm_orig = np.array(intersections_sm_orig)
                dist_sm_orig = np.linalg.norm(ret[0] - intersections_sm_orig[0])
                dist_sm_orig += np.linalg.norm(ret[1] - intersections_sm_orig[1])
                dist_sm_orig += np.linalg.norm(ret[2] - intersections_sm_orig[2])
                dist_sm_orig += np.linalg.norm(ret[3] - intersections_sm_orig[3])

                dif_sm_orig = abs(area_ret_sm_orig - area_sm_orig)

                '''print('Area Single Markers orig: ', area_sm_orig)
                print('Dif abs:', dif_sm_orig))
                print('Distancia:', dist_sm_orig)
                '''

                # Intersection Single Markers rot RED
                intersections_sm_rot2d = []
                for p in intersections_sm_rot:
                    aux = np.dot(cameraMatrix, p) / p[2]
                    aux_woZ = np.delete(aux, -1, axis=0)
                    intersections_sm_rot2d.append(aux_woZ)
                    ##print(aux_woZ)
                    #cv2.circle(original_image, tuple(np.int32(aux_woZ)), radius=20, color=(0, 0, 255), thickness=-1)

                intersections_sm_rot2d = np.asarray(intersections_sm_rot2d)
                
                area_triangle1 = triangle_area(intersections_sm_rot[0], intersections_sm_rot[1], intersections_sm_rot[2])
                area_triangle2 = triangle_area(intersections_sm_rot[0], intersections_sm_rot[3], intersections_sm_rot[2])
                area_sm_rot = (area_triangle1 + area_triangle2)/100

                # RET with SingleMarkers rot YELLOW
                ret = getCornersInCameraWorld(markerLength, rvec_sm_rot, [[tvec_sm_rot]])
                area_triangle1 = triangle_area(ret[0], ret[1], ret[2])
                area_triangle2 = triangle_area(ret[0], ret[2], ret[3])
                area_ret_sm_rot = (area_triangle1 + area_triangle2)/100
                #print('\nArea IDEAL(Single Markers rot): ', area_ret_sm_rot)

                aux = np.dot(cameraMatrix, tvec_sm_rot) / tvec_sm_rot[2]
                aux_woZ = np.delete(aux, -1, axis=0)
                #cv2.circle(original_image, tuple(np.int32(aux_woZ)), radius=20, color=(255, 0, 0), thickness=-1)

                for p in ret:
                    aux = np.dot(cameraMatrix, p) / p[2]
                    aux_woZ = np.delete(aux, -1, axis=0)
                    #cv2.circle(original_image, tuple(np.int32(aux_woZ)), radius=20, color=(0, 191, 255), thickness=-1)

                #intersections_sm_rot= np.array(intersections_sm_rot)
                dist_sm_rot = np.linalg.norm(ret[0] - intersections_sm_rot_copy[0])
                dist_sm_rot += np.linalg.norm(ret[1] - intersections_sm_rot_copy[1])
                dist_sm_rot += np.linalg.norm(ret[2] - intersections_sm_rot_copy[2])
                dist_sm_rot += np.linalg.norm(ret[3] - intersections_sm_rot_copy[3])

                dif_sm_rot = abs(area_ret_sm_rot - area_sm_rot)

                #Calcular com os pontos 2d reprojetados 
                '''print('Area Single Markers rot: ', area_sm_rot)
                print('Dif abs:', dif_sm_rot)
                print('distancia:', dist_sm_rot)
                '''
                
                # Intersection SolvePnP orig CYAN
                intersections_solve_orig2d = []
                for p in intersections_solve_orig:
                    aux = np.dot(cameraMatrix, p) / p[2]
                    aux_woZ = np.delete(aux, -1, axis=0)
                    intersections_solve_orig2d.append(aux_woZ)
                    #cv2.circle(original_image, tuple(np.int32(aux_woZ)), radius=20, color=(255, 255, 0), thickness=-1)

                intersections_solve_orig2d = np.asarray(intersections_solve_orig2d)
                area_triangle1 = triangle_area(intersections_solve_orig[0], intersections_solve_orig[1], intersections_solve_orig[2])
                area_triangle2 = triangle_area(intersections_solve_orig[0], intersections_solve_orig[2], intersections_solve_orig[3])
                area_solve_orig = (area_triangle1 + area_triangle2)/100

                # RET with solvePnP orig WHITE
                tvec_solve_orig = tvec_solve_orig.flatten()
                rvec_solve_orig = rvec_solve_orig.flatten()

                ret = getCornersInCameraWorld(markerLength, rvec_solve_orig, [[tvec_solve_orig]])
                area_triangle1 = triangle_area(ret[0], ret[1], ret[2])
                area_triangle2 = triangle_area(ret[0], ret[2], ret[3])
                area_ret_solve_orig = (area_triangle1 + area_triangle2)/100
                #print('\nArea IDEAL(Solve PnP orig): ', area_ret_solve_orig)
                
                aux = np.dot(cameraMatrix, tvec_solve_orig) / tvec_solve_orig[2]
                aux_woZ = np.delete(aux, -1, axis=0)
                #cv2.circle(original_image, tuple(np.int32(aux_woZ)), radius=20, color=(255, 0, 0), thickness=-1)
                
                for p in ret:
                    aux = np.dot(cameraMatrix, p) / p[2]
                    aux_woZ = np.delete(aux, -1, axis=0)
                    #cv2.circle(original_image, tuple(np.int32(aux_woZ)), radius=20, color=(255, 255, 255), thickness=-1)

                intersections_solve_orig = np.array(intersections_solve_orig)
                dist_solve_orig = np.linalg.norm(ret[0] - intersections_solve_orig[0])
                dist_solve_orig += np.linalg.norm(ret[1] - intersections_solve_orig[1])
                dist_solve_orig += np.linalg.norm(ret[2] - intersections_solve_orig[2])
                dist_solve_orig += np.linalg.norm(ret[3] - intersections_solve_orig[3])

                dif_solve_orig = abs(area_ret_solve_orig - area_solve_orig)

                '''print('Area solvePnP orig: ', area_solve_orig)
                print('Dif abs:', dif_solve_orig)
                print('Distancia:', dist_solve_orig)
                '''
                
                # Intersection SolvePnP rot ORANGE
                intersections_solve_rot2d = []
                for p in intersections_solve_rot:
                    aux = np.dot(cameraMatrix, p) / p[2]
                    aux_woZ = np.delete(aux, -1, axis=0)
                    intersections_solve_rot2d.append(aux_woZ)
                    #cv2.circle(original_image, tuple(np.int32(aux_woZ)), radius=20, color=(0, 140, 255), thickness=-1)

                intersections_solve_rot2d = np.asarray(intersections_solve_rot2d)
                #cv2.polylines(original_image, np.int32([intersections_sm_orig2d]), True, (0, 0, 255), 20) 

                area_triangle1 = triangle_area(intersections_solve_rot[0], intersections_solve_rot[1], intersections_solve_rot[2])
                area_triangle2 = triangle_area(intersections_solve_rot[0], intersections_solve_rot[2], intersections_solve_rot[3])
                area_solve_rot = (area_triangle1 + area_triangle2)/100
                
                # RET with solvePnP rot BLACK
                tvec_solve_rot = tvec_solve_rot.flatten()
                rvec_solve_rot = rvec_solve_rot.flatten()

                ret = getCornersInCameraWorld(markerLength, rvec_solve_rot, [[tvec_solve_rot]])
                area_triangle1 = triangle_area(ret[0], ret[1], ret[2])
                area_triangle2 = triangle_area(ret[0], ret[2], ret[3])
                area_ret_solve_rot = (area_triangle1 + area_triangle2)/100
                #print('\nArea IDEAL(Solve PnP rot): ', area_ret_solve_rot)
                
                aux = np.dot(cameraMatrix, tvec_solve_rot) / tvec_solve_rot[2]
                aux_woZ = np.delete(aux, -1, axis=0)
                #cv2.circle(original_image, tuple(np.int32(aux_woZ)), radius=20, color=(255, 0, 0), thickness=-1)
                
                for p in ret:
                    aux = np.dot(cameraMatrix, p) / p[2]
                    aux_woZ = np.delete(aux, -1, axis=0)
                    #cv2.circle(original_image, tuple(np.int32(aux_woZ)), radius=20, color=(0,0,0), thickness=-1)

                #intersections_solve_rot = np.array(intersections_solve_rot)
                dist_solve_rot = np.linalg.norm(ret[0] - intersections_solve_rot_copy[0])
                dist_solve_rot += np.linalg.norm(ret[1] - intersections_solve_rot_copy[1])
                dist_solve_rot += np.linalg.norm(ret[2] - intersections_solve_rot_copy[2])
                dist_solve_rot += np.linalg.norm(ret[3] - intersections_solve_rot_copy[3])

                dif_solve_rot = abs(area_ret_solve_rot - area_solve_rot)

                '''print('Area solvePnP rot: ', area_solve_rot)
                print('Dif abs:', dif_solve_rot)
                print('distancia:', dist_solve_rot)
                '''
                
                if(ordem == 0):
                        best_area = area_sm_orig
            
                if( (dif_sm_rot < dif_sm_orig) & 
                    (dif_solve_rot < dif_solve_orig) &
                    (abs(25 -  area_solve_rot) < 1) &
                    (abs(25 -  area_sm_rot) < 1) &
                    (dif_sm_rot < 1) &
                    (dif_solve_rot < 1) &
                    (abs(25 - area_sm_rot) < abs(25 - best_area)) &
                    (abs(25 - area_solve_rot) < abs(25 - best_area))
                    ):
                        
                    best_rotation = rotation
                    best_area = area_sm_rot

                    '''
                    print('Ordem: ', rotation)
                    print('Area solve orig', area_solve_orig)
                    print('Area solve rot', area_solve_rot)
                    print('Area sm orig', area_sm_orig)
                    print('Area sm rot', area_sm_rot)
                    print('Dist solve orig', dist_solve_orig)
                    print('Dist solve rot', dist_solve_rot)
                    print('Dist sm orig', dist_sm_orig)
                    print('Dist sm rot', dist_sm_rot, '\n')

                    img_copy = original_image.copy()
                    aruco.drawAxis(img_copy, cameraMatrix, dist, rvec_sm_orig, tvec_sm_orig, markerLength)  # Draw axis
                    cv2.imwrite(fname.replace(".png", "_axis_sm_orig_" + rotation + ".jpg"), img_copy)

                    img_copy = original_image.copy()
                    aruco.drawAxis(img_copy, cameraMatrix, dist, rvec_sm_rot, tvec_sm_rot, markerLength)  # Draw axis
                    cv2.imwrite(fname.replace(".png", "_axis_sm_rot_" + rotation + ".jpg"), img_copy)

                    cv2.imwrite(fname.replace(".png", "_points_" + rotation + ".jpg"), original_image)
                    '''
            print('Melhor rotation: ', best_rotation, '\nMelhor area: ', best_area)
            cv2.imwrite(newFname.replace(".jpg", "_" + best_rotation + ".jpg"), original_image)


import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from os import path

# Parameters
height1 = 4624
width1 = 3468
height2 = 512
width2 = 512

markerLength = 50

# Define scales
scale_x = width2 / width1
scale_y = width2 / width1

cameraMatrix_path = "../Original camera matrix.npy"
camera_matrix = np.load(cameraMatrix_path)

origin_file = "../new_origin_images.txt"

origin_values = {}
with open(origin_file, 'r') as file:
    for line in file:
        if line.strip():
            key, value = line.strip().split(':')
            key = key.strip().strip("'")
            value = int(value.strip().strip(','))
            origin_values[key] = value

def getCornersInCameraWorld(markerLength, rvec, tvec):
    objp = np.array([[-markerLength/2, markerLength/2, 0],
                     [markerLength/2, markerLength/2, 0],
                     [markerLength/2, -markerLength/2, 0],
                     [-markerLength/2, -markerLength/2, 0]], dtype=np.float32)

    R, _ = cv2.Rodrigues(rvec)
    corners = np.dot(R, objp.T).T + tvec
    return corners

def show_image(image, title):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for subelem in elem:
            indent(subelem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def format_matrix_data(matrix):
    formatted_lines = []
    for row in matrix:
        line = ' '.join(f"{item:.8e}" for item in row)
        formatted_lines.append(f"        {line}")
    aux = '\n'.join(formatted_lines)
    return '\n' + aux + '\n\t'

def create_and_save_xml(filename, leaf_number, image_size, new_cameraMatrix, homogeneous_mat, projection_mat_resized, output_path):
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "filename").text = filename

    # Read image size
    image_size_element = ET.SubElement(annotation, "image-size")
    ET.SubElement(image_size_element, "width").text = str(image_size[0])
    ET.SubElement(image_size_element, "height").text = str(image_size[1])
    ET.SubElement(image_size_element, "depth").text = str(image_size[2])

    # Read leaf number
    objects = ET.SubElement(annotation, "objects")
    ET.SubElement(objects, "leaf-number").text = str(leaf_number)

    # Read camera matrix
    calibration = ET.SubElement(annotation, "calibration")
    camera_matrix = ET.SubElement(calibration, "camera-matrix")
    ET.SubElement(camera_matrix, "rows").text = "3"
    ET.SubElement(camera_matrix, "cols").text = "3"
    ET.SubElement(camera_matrix, "data").text = format_matrix_data(new_cameraMatrix)

    # Read extrinsc homogeneous matrix
    extrinsic_homogeneous_matrix = ET.SubElement(calibration, "extrinsic-homogeneous-matrix")
    ET.SubElement(extrinsic_homogeneous_matrix, "rows").text = "4"
    ET.SubElement(extrinsic_homogeneous_matrix, "cols").text = "4"
    ET.SubElement(extrinsic_homogeneous_matrix, "data").text = format_matrix_data(np.array(homogeneous_mat))

    # Read model projection matrix
    model_projection_matrix = ET.SubElement(calibration, "model-projection-matrix")
    ET.SubElement(model_projection_matrix, "rows").text = "3"
    ET.SubElement(model_projection_matrix, "cols").text = "4"
    ET.SubElement(model_projection_matrix, "data").text = format_matrix_data(np.array(projection_mat_resized))

    # Indent and save XML
    indent(annotation)
    tree = ET.ElementTree(annotation)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

# Process leaves from 1 to n
n = 612  # Number of the last leaf to be processed
for i in range(1, n+1):
    leaf = str(i)
    leaf_filename = f"../Bean leaf dataset/{leaf.zfill(3)}"

    print(f"\nProcessing leaf {leaf}")

    leaves = {}
    with open(path.join(leaf_filename, 'nomes_arquivos.txt'), 'r') as arquivo:
        for linha in arquivo:
            chave, valor = linha.strip().split(' -> ')
            leaves[valor] = chave

    resized_folder = os.path.join(leaf_filename, "resized")
    annotation_folder = os.path.join(leaf_filename, "annotation")
    annotation_resized_folder = os.path.join(leaf_filename, "annotation_resized")

    # Create 'annotation_resized' folder if it doesn't exist
    if not os.path.exists(annotation_resized_folder):
        os.makedirs(annotation_resized_folder)

    for image_filename in os.listdir(resized_folder):
        if image_filename.endswith(".jpg"):
            print(f"\nProcessing image {image_filename} of leaf {leaf}")

            image_resize_filename = os.path.join(resized_folder, image_filename)
            annotation_filename = os.path.join(annotation_folder, image_filename.replace(".jpg", ".xml"))
            image_path = os.path.join(leaf_filename, image_filename)

            origin = origin_values[str(leaves[image_filename.replace('.jpg','')]) + '.jpg']
            
            new_cameraMatrix = camera_matrix.copy()
            new_cameraMatrix[0, 0] *= scale_x
            new_cameraMatrix[1, 1] *= scale_y
            new_cameraMatrix[0, 2] *= scale_x
            new_cameraMatrix[1, 2] = (new_cameraMatrix[1, 2] - origin) * scale_y

            tree = ET.parse(annotation_filename)
            root_xml = tree.getroot()

            image_resize = cv2.imread(image_resize_filename)
            image = cv2.imread(image_path)

            tvec = [float(val) for val in root_xml.find('./calibration/tvec').text.strip().split()]
            rvec = [float(val) for val in root_xml.find('./calibration/rvec').text.strip().split()]

            homogeneous_mat = []
            for row in root_xml.find('./calibration/extrinsic-homogeneous-matrix/data').text.strip().split('\n'):
                row_data = [float(val) for val in row.split()]
                homogeneous_mat.append(row_data)

            projection_mat = []
            for row in root_xml.find('./calibration/model-projection-matrix/data').text.strip().split('\n'):
                row_data = [float(val) for val in row.split()]
                projection_mat.append(row_data)

            rotation_matrix = []
            for row in root_xml.find('./calibration/rotation-matrix/data').text.strip().split('\n'):
                row_data = [float(val) for val in row.split()]
                rotation_matrix.append(row_data)

            rvec = np.array(rvec)
            tvec = np.array(tvec)

            Rt = np.hstack((rotation_matrix, tvec.reshape(3, 1)))

            projection_mat_resized = np.dot(new_cameraMatrix, Rt)

            teste_proj = np.dot(camera_matrix, Rt)

            corners = getCornersInCameraWorld(50, rvec, tvec)
            result_corners = []

            for corner in corners:
                result_corners.append(np.dot(new_cameraMatrix, corner) / corner[2])

            print(f"tvec: {tvec}")
            print(f"rvec: {rvec}")
            print(f"Extrinsic matrix: {homogeneous_mat}")
            print(f"Projection matrix: {projection_mat}")
            print(f"Resized projection matrix: {projection_mat_resized}")

            # Save XML file
            output_path = os.path.join(annotation_resized_folder, image_filename.replace(".jpg", ".xml"))
            create_and_save_xml(image_filename, leaf, (width2, height2, 3), new_cameraMatrix, homogeneous_mat, projection_mat_resized, output_path)
            
            '''
            # Draw and show images
            tvec_proj = np.dot(camera_matrix, tvec) / tvec[2]
            x1, y1 = int(tvec_proj[0]), int(tvec_proj[1])

            cv2.circle(image, (x1, y1), radius=50, color=(0, 0, 255), thickness=-1)

            for corner in corners:
                c = np.dot(camera_matrix, corner) / corner[2]
                x1, y1 = int(c[0]), int(c[1])
                cv2.circle(image, (x1, y1), radius=50, color=(0, 255, 0), thickness=-1)

            show_image(image, f"Imagem Original - Folha {leaf}")

            result = np.dot(new_cameraMatrix, tvec) / tvec[2]

            x2, y2 = int(result[0]), int(result[1])

            print(f"Projeção de tvec na imagem redimensionada: x2={x2}, y2={y2}")

            cv2.circle(image_resize, (x2, y2), radius=5, color=(0, 0, 255), thickness=-1)

            for corner in result_corners:
                x2, y2 = int(corner[0]), int(corner[1])
                cv2.circle(image_resize, (x2, y2), radius=5, color=(0, 255, 0), thickness=-1)

            show_image(image_resize, f"Imagem Redimensionada - Folha {leaf}")'''

print("Processing Completed!")

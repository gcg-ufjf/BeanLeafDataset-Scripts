import re
import pandas as pd
from bs4 import BeautifulSoup
from os import listdir, path, makedirs
from glob import glob
from new_origin_images_dict import imgs_dict as origins_dict
import numpy as np

dataset_path = "./Bean leaf dataset/"
output_path = "./Bean leaf dataset2/"
sheet_path = ".../Bean_Leaf_Dataset_Infos.xlsx"

# Template for the new annotation format
template = """\
<annotation>
  <image-name>{image_name}</image-name>
  <observation>{observation}</observation>
  <image-size>
    <height>512</height>
    <width>512</width>
    <depth>3</depth>
  </image-size>
  <objects>
    <leaf-number>{leaf_number}</leaf-number>
    <marker>
      <marker-real-area>25</marker-real-area>
      <marker-projected-area>{projected_area}</marker-projected-area>
      <normalized-corners>
{corners}
      </normalized-corners>
    </marker>
    <leaf>
      <dimensions>
        <leaf-area>{area}</leaf-area>
        <leaf-width>{width}</leaf-width>
        <leaf-length>{length}</leaf-length>
        <leaf-perimeter>{perimeter}</leaf-perimeter>
      </dimensions>
      <normalized-polygon>
{polygon}
      </normalized-polygon>
    </leaf>
  </objects>
  <calibration>
    <camera-matrix>
      <rows>3</rows>
      <cols>3</cols>
      <data>
        {camera_matrix_data}
      </data>
    </camera-matrix>
    <tvec>
      {tvec}
    </tvec>
    <rvec >
      {rvec}
    </rvec>
    <rotation-matrix>
      <rows>3</rows>
      <cols>3</cols>
      <data>
        {rotation_matrix_data}
      </data>
    </rotation-matrix>
    <extrinsic-homogeneous-matrix>
      <rows>4</rows>
      <cols>4</cols>
      <data>
        {extrinsic_homogeneous_matrix_data}
      </data>
    </extrinsic-homogeneous-matrix>
    <model-projection-matrix>
      <rows>3</rows>
      <cols>4</cols>
      <data>
        {model_projection_matrix_data}
      </data>
    </model-projection-matrix>
  </calibration>
</annotation>\
"""

def find(root: BeautifulSoup, tag_name: str) -> str:
    """
    Get a BeautifulSoup xml file and return the value(s) of the corresponding tag as a string
    Tags can be indexed in the following format: tag1.otherTag.[...].lastTag
    """

    tags = tag_name.split(".")
    el = root
    for tag_name in tags:
        el = el.find(tag_name)
        if el is None:
            return None

    return "\n".join(filter(
        lambda c: c != '',
        [str(x).strip() for x in el.children]
    ))

def renormalize(points: str, origin) -> str:
    """ Normalize annotation points to 512x512 resolution and identify them in the file. """

    regex = r'(0\.\d+)'
    points = points.split("\n")

    for i in range(0, len(points), 2):
        x = float(re.search(regex, points[i]).group(1))
        y = float(re.search(regex, points[i + 1]).group(1))

        x = (x * 4624) / 3468
        y = ((y * 4624) - origin) / 3468

        points[i] = " " * 8 + re.sub(regex, str(x), points[i])
        points[i + 1] = " " * 10 + re.sub(regex, str(y), points[i + 1])
    return "\n".join(points)

def ident_items(items: str) -> str:
    return "\n".join(map(lambda i: " " * 6 + i, items.split("\n")))

if __name__ == "__main__":
    df = pd.read_excel(sheet_path)
    for dir in listdir(dataset_path): 
      try:
          print(dir)

          out_path = path.join(output_path, dir,"annotation.")
          makedirs(out_path, exist_ok=True)
          
          notes = df[df["Leaf"] == int(dir)]["Notes"].iloc[0]
          notes = ";".join([note.strip() for note in notes.split(";")])
          annotations = glob(path.join(dataset_path, dir, "annotation_resized", "*.xml"))
          annotations.sort()

          leaves = {}
          with open(path.join(dataset_path, dir,'nomes_arquivos.txt'), 'r') as arquivo:
            for linha in arquivo:
              chave, valor = linha.strip().split(' -> ')
              leaves[valor] = chave

          for leaf_m, xml_path in enumerate(annotations):
              file_name = path.basename(xml_path)
              print("-", file_name)

              marker_area_path = xml_path.replace('annotation_resized','area_estimation/marker').replace('.xml', '_area.raw')
              print("-", marker_area_path)

              proj_area = np.sum(np.fromfile(marker_area_path, dtype=np.double))
              print("-", proj_area)

              # Read old xml
              with open(xml_path) as file:
                  content = "".join(file.readlines())
                  old_xml = BeautifulSoup(content, "xml")

              new_name = file_name.split(".xml")[0] + ".jpg"
              old_file_name = leaves.get(file_name.split(".xml")[0], None)
              
              # Reads annotations and updates normalization to new size
              new_origin = origins_dict[old_file_name]

              if find(old_xml, "normalized-rotated-corners") is not None:
                corners = find(old_xml, "normalized-rotated-corners")
              else:
                 corners = find(old_xml, "normalized-original-corners")

              corners = renormalize(corners, new_origin)
              polygon = find(old_xml, "leaf.normalized-polygon")
              polygon = renormalize(polygon, new_origin)

              # Create a dictionary with the values that will compose the new annotation file
              values = {
                  "image_name": new_name,
                  "observation": notes,
                  "leaf_number": find(old_xml, "leaf-number"),
                  "projected_area": proj_area,
                  "corners": corners,
                  "area": find(old_xml, "leaf.area"),
                  "width": find(old_xml, "leaf.width"),
                  "length": find(old_xml, "leaf.length"),
                  "perimeter": find(old_xml, "leaf.perimeter"),
                  "polygon": polygon,
                  "camera_matrix_data": find(old_xml, "camera-matrix.data"),
                  "coefficients": ident_items(find(old_xml, "distortion-coefficients")),
                  "tvec": find(old_xml, "tvec"),
                  "rvec": find(old_xml, "rvec"),
                  "rotation_matrix_data":  find(old_xml, "rotation-matrix.data"),
                  "extrinsic_homogeneous_matrix_data": find(old_xml, "extrinsic-homogeneous-matrix.data"),
                  "model_projection_matrix_data": find(old_xml, "model-projection-matrix.data"),
              }

              with open(path.join(out_path, file_name), "w") as out_file:
                  out_file.write(template.format(**values))
      except:
        pass

import re
import os
import glob

def substitute_multiple_tags(xml_file1, xml_file2, output_file, tags):
    # Read the first XML file
    with open(xml_file1, 'r') as file:
        xml_content1 = file.read()

    # Read the second XML file
    with open(xml_file2, 'r') as file:
        xml_content2 = file.read()

    # Iterate over each tag and substitute its <data> content
    for tag_name in tags:
        pattern = re.compile(rf'<{tag_name}>\s*<rows>\s*\d+\s*</rows>\s*<cols>\s*\d+\s*</cols>\s*<data>(.*?)</data>\s*</{tag_name}>', re.DOTALL)
        match2 = pattern.search(xml_content2)
        if not match2:
            raise ValueError(f"Tag <{tag_name}> or its <data> content not found in the second XML")

        new_data_content = match2.group(1).strip()

        # Replace the <data> content in the first XML with the content from the second XML
        def replace_data(match):
            old_data = match.group(1).strip()
            return match.group(0).replace(old_data, new_data_content)

        # Update the content in the first XML for the current tag
        xml_content1 = pattern.sub(replace_data, xml_content1)

    # Write the updated content to the output file
    with open(output_file, 'w') as file:
        file.write(xml_content1)

# Define the base path for the folders
base_path = '../Bean leaf dataset/'
tags = ['camera-matrix', 'extrinsic-homogeneous-matrix', 'model-projection-matrix']

# Loop through leaves 001 to 612
for i in range(1, 613):
    folder_name = f'{i:03}'  # Format folder number
    annotation_folder = os.path.join(base_path, folder_name, 'annotation')
    annotation_resized_folder = os.path.join(base_path, folder_name, 'annotation_resized')

    # Find XML files
    xml_files1 = glob.glob(os.path.join(annotation_folder, '*.xml'))
    xml_files2 = glob.glob(os.path.join(annotation_resized_folder, '*.xml'))

    # Ensure we have the same number of XML files in both directories
    if len(xml_files1) != len(xml_files2):
        print(f"Warning: Different number of XML files in {annotation_folder} and {annotation_resized_folder}")
        continue

    # Sort files to ensure they match correctly by name
    xml_files1.sort()
    xml_files2.sort()

    # Loop over all XML files and substitute data
    for xml_file1, xml_file2 in zip(xml_files1, xml_files2):
        output_file = xml_file2  # Writing the updated content to the resized folder

        try:
            substitute_multiple_tags(xml_file1, xml_file2, output_file, tags)
            print(f"Updated {output_file} successfully.")
        except Exception as e:
            print(f"Error processing {xml_file1} and {xml_file2}: {e}")
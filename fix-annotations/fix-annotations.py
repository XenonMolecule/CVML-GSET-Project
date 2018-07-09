import sys
import os

# Get the annotation folder path as argument
annotation_path = sys.argv[1]
# Get the target directory path as argument
replacement_path = sys.argv[2]

# Add slash to the end of the replacement path if forgotten
if(replacement_path[-1] != "/"):
    replacement_path = replacement_path + "/"

annotation_dir = os.fsencode(annotation_path)
replacement_dir = os.fsencode(replacement_path)

for file in os.listdir(annotation_dir):
    filename = os.fsdecode(file)
    file = open(annotation_path + filename, "r")
    file_string = file.read()

    # Find the folder of the image
    start = file_string.find("<folder>")
    end = file_string.find("</folder>")
    folder = file_string[start+8:end]

    # Find the filename of the image
    start = file_string.find("<filename>")
    end = file_string.find("</filename>")
    img_name = file_string[start+10:end]

    # Replace the path with the provided path from arg
    start = file_string.find("<path>")
    end = file_string.find("</path>")
    old_path = file_string[start+6:end]
    new_path = replacement_path + folder + "/" + img_name
    file_string = file_string.replace(old_path, new_path)

    file.close()

    # Actually Write to the File
    file_w = open(annotation_path + filename, "w")
    file_w.write(file_string)
    file_w.close()

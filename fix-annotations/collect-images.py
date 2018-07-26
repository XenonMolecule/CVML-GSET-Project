import sys
import os
from shutil import copyfile
from random import randint
import xml.etree.ElementTree as ET

ANNOTATION_DIR = "C:\\Users\\micha\\OneDrive\\Documents\\GitHub\\CVML-GSET-Project\\ensemble\\evaluation\\ground-truth\\backup\\"
TARGET_IMG_DIR = "C:\\Users\\micha\\OneDrive\\Documents\\GitHub\\CVML-GSET-Project\\ensemble\\evaluation\\test-images\\"

annotations = os.fsencode(ANNOTATION_DIR)
for file in os.listdir(annotations) :
    filename = os.fsdecode(file)
    xml_file = ET.parse(ANNOTATION_DIR + filename).getroot()
    im_filename = xml_file.find('filename').text
    xml_file = ET.parse(ANNOTATION_DIR + filename).getroot()
    im_path = xml_file.find('path').text
    copyfile(im_path, TARGET_IMG_DIR + im_filename)

import sys
import os
from shutil import copyfile
from random import randint

ANNOTATION_DIR = ""
TRAIN_IMG_DIR = ""

annotations = os.fsencode(ANNOTATION_PATH)
for (file in os.listdir(annotations)):
    filename = os.fsencode(file)
    copyfile(ANNOTATION_DIR + filename, TRAIN_IMG_DIR + filename)

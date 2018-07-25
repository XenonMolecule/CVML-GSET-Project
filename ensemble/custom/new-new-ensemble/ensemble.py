from prediction import Box
from prediction import Prediction

from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import sys

import rfcn
import classifier
import detector2

PATH_TO_IMAGE = 'C:/Users/micha/OneDrive/Documents/GitHub/CVML-GSET-Project/dataset/test-images/test1.jpg'
PATH_TO_LABELS = 'C:/Users/micha/OneDrive/Documents/GitHub/CVML-GSET-Project/dataset/new-tensorflow-dataset/data/label_map.pbtxt'
NUM_CLASSES = 8
IMAGE_SIZE = (12, 8)

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("C:\\Users\\micha\\OneDrive\\Documents\\GitHub\\models\\research")
sys.path.append("C:\\Users\\micha\\OneDrive\\Documents\\GitHub\\models\\research\\object_detection")
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from utils import label_map_util
from utils import visualization_utils as vis_util

# Load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def display_prediction(image, prediction):
    vis_util.visualize_boxes_and_labels_on_image_array(image, np.array(prediction.get_coords()),
      prediction.get_class_labels(), prediction.get_confidences(),
      category_index, use_normalized_coordinates=True,
      line_thickness=8)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image)
    plt.show()

final_prediction = Prediction()

image = cv2.cvtColor(cv2.imread(PATH_TO_IMAGE), cv2.COLOR_BGR2RGB)
prediction_one = rfcn.predict_img(image, False)
for box in prediction_one.get_boxes():
    new_img = box.splice_img(image)
    prediction_n = detector2.predict_img(new_img, False)
    matches = 0
    confidences = []
    for mini_box in prediction_n.get_boxes():
        class_prediction = classifier.predict_img(mini_box.splice_img(new_img), False)
        confidences.append(class_prediction.get_confidence())
        if(mini_box.class_label == class_prediction.class_label):
            matches += 1
    if(matches == len(prediction_n.get_boxes())):
        i = 0
        for mini_box in prediction_n.get_boxes():
            box_coords = box.get_coordinates_absolute(image)
            mini_box_coords = mini_box.get_coordinates_absolute(new_img)
            im_height = image.shape[0]
            im_width = image.shape[1]
            true_box = Box((box_coords[0] + mini_box_coords[0])/im_height, (box_coords[1] + mini_box_coords[1])/im_width,
              (box_coords[0] + mini_box_coords[2])/im_height, (box_coords[1] + mini_box_coords[3])/im_width, confidences[i], mini_box.get_class_label())
            final_prediction.append_box(true_box)
            i += 1
    else:
        final_prediction.append_box(box)

display_prediction(image, final_prediction)

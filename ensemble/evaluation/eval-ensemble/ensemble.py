import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from prediction import Box
from prediction import Prediction

import classifier

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("C:\\Users\\micha\\OneDrive\\Documents\\GitHub\\models\\research")
sys.path.append("C:\\Users\\micha\\OneDrive\\Documents\\GitHub\\models\\research\\object_detection")
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from utils import label_map_util
from utils import visualization_utils as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT_1 = 'C:/Users/micha/OneDrive/Documents/GitHub/CVML-GSET-Project/detector-models/rfcnmodel/models/rfcn/export/frozen_inference_graph.pb'
PATH_TO_CKPT_2 = 'C:/Users/micha/OneDrive/Documents/GitHub/CVML-GSET-Project/detector-models/ssdmodel/models/ssd/export/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'C:/Users/micha/OneDrive/Documents/GitHub/CVML-GSET-Project/dataset/new-tensorflow-dataset/data/label_map.pbtxt'
#IMG Path
PATH_TO_IMAGE = 'C:/Users/micha/OneDrive/Documents/GitHub/CVML-GSET-Project/dataset/test-images/test1.jpg'
IMG_DIR = "C:\\Users\\micha\\OneDrive\\Documents\\GitHub\\CVML-GSET-Project\\ensemble\\evaluation\\test-images\\"
OUTPUT_TXT_DIR = "C:\\Users\\micha\\OneDrive\\Documents\\GitHub\\CVML-GSET-Project\\ensemble\\evaluation\\predictions\\"

NUM_CLASSES = 8
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
CONFIDENCE_THRESH = 0.5

def conv_id_to_name(id):
    if(id == 1):
        return "bottle"
    elif(id == 2):
        return "can"
    elif(id == 3):
        return "container"
    elif(id == 4):
        return "wrapper"
    elif(id == 5):
        return "paper"
    elif(id == 6):
        return "cardboard"
    elif(id == 7):
        return "cup"
    elif(id == 8):
        return "scrap"

# Load tensorflow model
detection_graph1 = tf.Graph()
with detection_graph1.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT_1, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Load tensorflow model
detection_graph2 = tf.Graph()
with detection_graph2.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT_2, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def fix_output_dict(output_dict):
    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return conv_output_dict_to_prediction(output_dict)

def conv_output_dict_to_prediction(output_dict):
    prediction = Prediction()
    for i in range(len(output_dict['detection_boxes'])) :
        if(output_dict['detection_scores'][i] >= CONFIDENCE_THRESH):
            detection_boxes = (output_dict['detection_boxes'][i])
            new_box = Box(detection_boxes[0], detection_boxes[1], detection_boxes[2],
                detection_boxes[3], output_dict['detection_scores'][i], output_dict['detection_classes'][i])
            prediction.append_box(new_box)
    return prediction

def display_prediction(image, prediction):
    vis_util.visualize_boxes_and_labels_on_image_array(image, np.array(prediction.get_coords()),
      prediction.get_class_labels(), prediction.get_confidences(),
      category_index, use_normalized_coordinates=True,
      line_thickness=8)
    cv2.imshow('object detection', cv2.resize(image, (800, 600)))

def run_ensemble(graph1, graph2):
    # INITIALIZE DETECTOR 1
    with tf.Session(graph=graph1) as sess1:
        # Get handles to input and output tensors
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict1 = {}
        for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict1[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
        if 'detection_masks' in tensor_dict1:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict1['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict1['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict1['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict1['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
        image_tensor1 = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        # INITIALIZE DETECTOR 2
        with tf.Session(graph=graph2) as sess2:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict2 = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict2[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict2:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict2['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict2['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict2['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict2['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            image_tensor2 = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # ACTUALLY RUN THE ENSEMBLER
            for file in os.listdir(os.fsencode(IMG_DIR)):
                filename = os.fsdecode(file)
                image = cv2.imread(IMG_DIR + filename)

                final_prediction = Prediction()

                # Run inference
                prediction_1 = fix_output_dict(sess1.run(tensor_dict1, feed_dict={image_tensor1: np.expand_dims(image, 0)}))

                for box in prediction_1.get_boxes():
                    new_img = box.splice_img(image)
                    prediction_n = fix_output_dict(sess2.run(tensor_dict2, feed_dict={image_tensor2: np.expand_dims(new_img, 0)}))
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
                filename = filename.split(".jpg")[0]
                filename = filename.split(".png")[0]
                filename = filename.split(".jpeg")[0]
                output_file = open(OUTPUT_TXT_DIR + filename + ".txt", "w+")
                for box in final_prediction.get_boxes():
                    abs_coords = box.get_coordinates_absolute(image)
                    class_label = conv_id_to_name(box.get_class_label())
                    confidence = box.get_confidence()
                    output_file.write(class_label + " " + str(confidence) + " " +
                    str(abs_coords[1]) + " " + str(abs_coords[0]) + " " +
                    str(abs_coords[3]) + " " + str(abs_coords[2]) + "\n")
                output_file.close()

                if(cv2.waitKey(25) & 0xFF == ord('q')):
                    cv2.destroyAllWindows()
                    break

run_ensemble(detection_graph1, detection_graph2)

def process_img(image_np, display):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.
    if(display):
        vis_util.visualize_boxes_and_labels_on_image_array(image_np, output_dict['detection_boxes'],
          output_dict['detection_classes'], output_dict['detection_scores'],
          category_index, instance_masks=output_dict.get('detection_masks'),
          use_normalized_coordinates=True, line_thickness=8)
        plt.figure(figsize=IMAGE_SIZE)
        plt.imshow(image_np)
        plt.show()

    return output_dict

def predict_img(image, display):
    output_dict = process_img(image, display)
    prediction = Prediction()
    for i in range(len(output_dict['detection_boxes'])) :
        if(output_dict['detection_scores'][i] >= CONFIDENCE_THRESH):
            detection_boxes = (output_dict['detection_boxes'][i])
            new_box = Box(detection_boxes[0], detection_boxes[1], detection_boxes[2],
                detection_boxes[3], output_dict['detection_scores'][i], output_dict['detection_classes'][i])
            prediction.append_box(new_box)
    return prediction

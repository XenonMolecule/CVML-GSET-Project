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

from time import sleep
from threading import Thread
import socket
from struct import unpack

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("C:\\Users\\micha\\OneDrive\\Documents\\GitHub\\models\\research")
sys.path.append("C:\\Users\\micha\\OneDrive\\Documents\\GitHub\\models\\research\\object_detection")
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from utils import label_map_util
from utils import visualization_utils as vis_util

images_to_process = []

class DroneVideoStream:
    def __init__(self):
        self.stopped = False

    def start(self):
		# start the thread to read frames from the video stream
        print("Looking for connections")
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        s.bind(('localhost', 9090))
        s.listen(1)

        client_socket, addr = s.accept()
        print("Got Connection")
        while not self.stopped:
            print("Running")
            buf = b''
            while len(buf)<4:
                buf += client_socket.recv(4-len(buf))
            size = unpack('!i', buf)
            img = b''
            if(size[0] > 0):
                img = client_socket.recv(size[0]+4)
            if(not size[0] > 30000 and not size[0] < 0):
                image = open('test.jpg', 'wb')
                image.write(img)
                image_np = cv2.imread('test.jpg')
                images_to_process.append(image_np)
                if(cv2.waitKey(1) & 0xFF == ord('q')):
                    cv2.destroyAllWindows()
                    break

        cv2.waitKey(1)

    def stop(self):
		# indicate that the thread should be stopped
        self.stopped = True

droneVS = DroneVideoStream()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT_1 = 'C:/Users/micha/OneDrive/Documents/GitHub/CVML-GSET-Project/detector-models/ssdmodel/models/ssd/export/frozen_inference_graph.pb'
PATH_TO_CKPT_2 = 'C:/Users/micha/OneDrive/Documents/GitHub/CVML-GSET-Project/detector-models/rfcnmodel/models/rfcn/export/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'C:/Users/micha/OneDrive/Documents/GitHub/CVML-GSET-Project/dataset/new-tensorflow-dataset/data/label_map.pbtxt'
#IMG Path
PATH_TO_IMAGE = 'C:/Users/micha/OneDrive/Documents/GitHub/CVML-GSET-Project/dataset/test-images/test1.jpg'
IMG_DIR = "C:\\Users\\micha\\OneDrive\\Documents\\GitHub\\CVML-GSET-Project\\ensemble\\evaluation\\test-images\\"
OUTPUT_TXT_DIR = "C:\\Users\\micha\\OneDrive\\Documents\\GitHub\\CVML-GSET-Project\\ensemble\\averager\\predicted\\"

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

def GeneralEnsemble(dets, iou_thresh = 0.5, weights=None):
    assert(type(iou_thresh) == float)

    ndets = len(dets)

    if weights is None:
        w = 1/float(ndets)
        weights = [w]*ndets
    else:
        assert(len(weights) == ndets)

        s = sum(weights)
        for i in range(0, len(weights)):
            weights[i] /= s

    out = list()
    used = list()

    for idet in range(0,ndets):
        det = dets[idet]
        for box in det:
            if box in used:
                continue

            used.append(box)
            # Search the other detectors for overlapping box of same class
            found = []
            for iodet in range(0, ndets):
                odet = dets[iodet]

                if odet == det:
                    continue

                bestbox = None
                bestiou = iou_thresh
                for obox in odet:
                    if not obox in used:
                        # Not already used
                        if box[4] == obox[4]:
                            # Same class
                            iou = computeIOU(box, obox)
                            if iou > bestiou:
                                bestiou = iou
                                bestbox = obox

                if not bestbox is None:
                    w = weights[iodet]
                    found.append((bestbox,w))
                    used.append(bestbox)

            # Now we've gone through all other detectors
            if len(found) == 0:
                new_box = list(box)
                new_box[5] /= ndets
                out.append(new_box)
            else:
                allboxes = [(box, weights[idet])]
                allboxes.extend(found)

                xc = 0.0
                yc = 0.0
                bw = 0.0
                bh = 0.0
                conf = 0.0

                wsum = 0.0
                for bb in allboxes:
                    w = bb[1]
                    wsum += w

                    b = bb[0]
                    xc += w*b[0]
                    yc += w*b[1]
                    bw += w*b[2]
                    bh += w*b[3]
                    conf += w*b[5]

                xc /= wsum
                yc /= wsum
                bw /= wsum
                bh /= wsum

                new_box = [xc, yc, bw, bh, box[4], conf]
                out.append(new_box)
    return out

def getCoords(box):
    x1 = float(box[0]) - float(box[2])/2
    x2 = float(box[0]) + float(box[2])/2
    y1 = float(box[1]) - float(box[3])/2
    y2 = float(box[1]) + float(box[3])/2
    return x1, x2, y1, y2

def computeIOU(box1, box2):
    x11, x12, y11, y12 = getCoords(box1)
    x21, x22, y21, y22 = getCoords(box2)

    x_left   = max(x11, x21)
    y_top    = max(y11, y21)
    x_right  = min(x12, x22)
    y_bottom = min(y12, y22)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersect_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)

    iou = intersect_area / (box1_area + box2_area - intersect_area)
    return iou

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

            droneVS.start()

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

            loop_count = 0


            # ACTUALLY RUN THE ENSEMBLER
            while True:
                if loop_count == 1:
                    images_to_process = []
                if(len(images_to_process) > 0):
                    image = images_to_process.pop(0)
                    # Purposefully drop frames to catch up
                    if(len(images_to_process) > 5):
                        image = images_to_process.pop(0)
                    if(len(images_to_process) > 10):
                        images_to_process.pop(0)
                        image = images_to_process.pop(0)
                    if(len(images_to_process) > 50):
                        image = images_to_process.pop(0)
                        images_to_process = []
                    if(type(image) == np.ndarray):

                        final_prediction = Prediction()

                        # Run inference
                        prediction_1 = fix_output_dict(sess1.run(tensor_dict1, feed_dict={image_tensor1: np.expand_dims(image, 0)}))
                        prediction_2 = fix_output_dict(sess2.run(tensor_dict2, feed_dict={image_tensor2: np.expand_dims(image, 0)}))

                        detections = []
                        indiv_detections = []

                        for box in prediction_1.get_boxes():
                            indiv_detections.append(box.conv_to_center_mode())
                        detections.append(indiv_detections)

                        indiv_detections = []
                        for box in prediction_2.get_boxes():
                            indiv_detections.append(box.conv_to_center_mode())
                        detections.append(indiv_detections)

                        new_detections = GeneralEnsemble(detections)

                        for box in new_detections:
                            gen_box = Box(0,0,0,0,0,0).create_from_center_mode(box)
                            final_prediction.append_box(gen_box)

                        display_prediction(image, final_prediction)

                        loop_count += 1

                        if(cv2.waitKey(25) & 0xFF == ord('q')):
                            cv2.destroyAllWindows()
                            break
                else:
                    sleep(0.05)

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

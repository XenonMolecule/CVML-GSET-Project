import socket
import cv2
import numpy as np
import os
from struct import unpack

import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from time import sleep
from threading import Thread

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
PATH_TO_CKPT = 'C:/Users/micha/OneDrive/Documents/\GitHub/CVML-GSET-Project/detector-models/rfcnmodel/models/rfcn/export/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'C:/Users/micha/OneDrive/Documents/\GitHub/CVML-GSET-Project/detector-models/rfcnmodel/data/label_map.pbtxt'

NUM_CLASSES = 8

# Load tensorflow model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

PATH_TO_TEST_IMAGES_DIR = 'C:/Users/micha/OneDrive/Documents/GitHub/CVML-GSET-Project/dataset/test-images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'test{}.jpg'.format(i)) for i in range(1, 4) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

graph = detection_graph
loop_count = 0
with graph.as_default():
    with tf.Session() as sess:

        droneVS.start()

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
                    # Get handles to input and output tensors
                    ops = tf.get_default_graph().get_operations()
                    all_tensor_names = {output.name for op in ops for output in op.outputs}
                    tensor_dict = {}
                    for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
                        tensor_name = key + ':0'
                        if tensor_name in all_tensor_names:
                            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                    if 'detection_masks' in tensor_dict:
                        # The following processing is only for single image
                        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
                        detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                        # Follow the convention by adding back the batch dimension
                        tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
                    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                    # Run inference
                    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

                    # all outputs are float32 numpy arrays, so convert types as appropriate
                    output_dict['num_detections'] = int(output_dict['num_detections'][0])
                    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                    output_dict['detection_scores'] = output_dict['detection_scores'][0]
                    if 'detection_masks' in output_dict:
                        output_dict['detection_masks'] = output_dict['detection_masks'][0]
                    vis_util.visualize_boxes_and_labels_on_image_array(image, output_dict['detection_boxes'],
                      output_dict['detection_classes'], output_dict['detection_scores'],
                      category_index, instance_masks=output_dict.get('detection_masks'),
                      use_normalized_coordinates=True, line_thickness=8)
                    cv2.imshow('object detection', cv2.resize(image, (800, 600)))
                    loop_count += 1
                    if(cv2.waitKey(25) & 0xFF == ord('q')):
                        cv2.destroyAllWindows()
                        break
                else:
                    sleep(0.05)

droneVS.stop()

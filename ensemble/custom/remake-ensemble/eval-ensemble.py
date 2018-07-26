# Copyright 2017 The TensorFlow Authors and Moobchooboobl. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Infers detections on a TFRecord of TFExamples given an inference graph.
Example usage:
  ./infer_detections \
    --input_tfrecord_paths=/path/to/input/tfrecord1,/path/to/input/tfrecord2 \
    --output_tfrecord_path_prefix=/path/to/output/detections.tfrecord \
    --inference_graph=/path/to/frozen_weights_inference_graph.pb
The output is a TFRecord of TFExamples. Each TFExample from the input is first
augmented with detections from the inference graph and then copied to the
output.
The input and output nodes of the inference graph are expected to have the same
types, shapes, and semantics, as the input and output nodes of graphs produced
by export_inference_graph.py, when run with --input_type=image_tensor.
The script can also discard the image pixels in the output. This greatly
reduces the output size and can potentially accelerate reading data in
subsequent processing steps that don't require the images (e.g. computing
metrics).
"""

import sys

sys.path.append("C:\\Users\\micha\\OneDrive\\Documents\\GitHub\\models\\research")
sys.path.append("C:\\Users\\micha\\OneDrive\\Documents\\GitHub\\models\\research\\object_detection")

import itertools
import tensorflow as tf
from object_detection.inference import detection_inference
import numpy as np

tf.flags.DEFINE_string('input_tfrecord_paths', None,
                       'A comma separated list of paths to input TFRecords.')
tf.flags.DEFINE_string('output_tfrecord_path', None,
                       'Path to the output TFRecord.')
tf.flags.DEFINE_boolean('discard_image_pixels', False,
                        'Discards the images in the output TFExamples. This'
                        ' significantly reduces the output size and is useful'
                        ' if the subsequent tools don\'t need access to the'
                        ' images (e.g. when computing evaluation measures).')

FLAGS = tf.flags.FLAGS

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT_1 = 'C:/Users/micha/OneDrive/Documents/GitHub/CVML-GSET-Project/detector-models/ssdmodel/models/ssd/export/frozen_inference_graph.pb'
PATH_TO_CKPT_2 = 'C:/Users/micha/OneDrive/Documents/GitHub/CVML-GSET-Project/detector-models/rfcnmodel/models/rfcn/export/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'C:/Users/micha/OneDrive/Documents/GitHub/CVML-GSET-Project/dataset/new-tensorflow-dataset/data/label_map.pbtxt'
#IMG Path
PATH_TO_IMAGE = 'C:/Users/micha/OneDrive/Documents/GitHub/CVML-GSET-Project/dataset/test-images/test1.jpg'

NUM_CLASSES = 8
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
CONFIDENCE_THRESH = 0.5

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

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  required_flags = ['input_tfrecord_paths', 'output_tfrecord_path']
  for flag_name in required_flags:
    if not getattr(FLAGS, flag_name):
      raise ValueError('Flag --{} is required'.format(flag_name))

  # INITIALIZE DETECTOR 1
  with tf.Session(graph=detection_graph1) as sess1:
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
    with tf.Session(graph=detection_graph2) as sess2:
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

        input_tfrecord_paths = [
            v for v in FLAGS.input_tfrecord_paths.split(',') if v]
        tf.logging.info('Reading input from %d files', len(input_tfrecord_paths))
        serialized_example_tensor, image_tensor = detection_inference.build_input(
            input_tfrecord_paths)

        tf.logging.info('Running inference and writing output to {}'.format(
            FLAGS.output_tfrecord_path))
        # sess.run(tf.local_variables_initializer())
        # tf.train.start_queue_runners()
        with tf.python_io.TFRecordWriter(
            FLAGS.output_tfrecord_path) as tf_record_writer:
          try:
            record_iterator = tf.python_io.tf_record_iterator(path=FLAGS.input_tfrecord_paths)
            for string_record in record_iterator:
              tf_example = tf.train.Example()
              tf_example.ParseFromString(string_record)
              tf.logging.info(tf_example.features.feature['image_raw'])
              height = int(tf_example.features.feature['height'].int64_list.value[0])
              width = int(tf_example.features.feature['width'].int64_list.value[0])
              img_string = (tf_example.features.feature['image_raw'].bytes_list.value[0])
              img_1d = np.fromstring(img_string, dtype=np.uint8)
              image = img_1d.reshape((height, width, -1))

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

              feature = tf_example.features.feature
              feature[standard_fields.TfExampleFields.
                      detection_score].float_list.value[:] = final_prediction.get_confidences()
              feature[standard_fields.TfExampleFields.
                      detection_bbox_ymin].float_list.value[:] = final_prediction.get_ymins()
              feature[standard_fields.TfExampleFields.
                      detection_bbox_xmin].float_list.value[:] = final_prediction.get_xmins()
              feature[standard_fields.TfExampleFields.
                      detection_bbox_ymax].float_list.value[:] = final_prediction.get_ymaxs()
              feature[standard_fields.TfExampleFields.
                      detection_bbox_xmax].float_list.value[:] = final_prediction.get_xmaxs()
              feature[standard_fields.TfExampleFields.
                      detection_class_label].int64_list.value[:] = final_prediction.get_class_labels()

              if FLAGS.discard_image_pixels:
                del feature[standard_fields.TfExampleFields.image_encoded]

              tf_record_writer.write(tf_example.SerializeToString())
          except tf.errors.OutOfRangeError:
            tf.logging.info('Finished processing records')
        tf.logging.info("Woot woot - done")


if __name__ == '__main__':
  tf.app.run()

import tensorflow as tf
import sys
import os
import xml.etree.ElementTree as ET

from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

annotation_path = "/c/Users/micha/OneDrive/Documents/GitHub/CVML-GSET-Project/dataset/general-dataset/trashnet-annotations/"
annotation_dir = os.fsencode(annotation_path)

def conv_name_to_id(name):
    if(name == "bottle"):
        return 1
    elif(name == "can"):
        return 2
    elif(name == "container"):
        return 3
    elif(name == "wrapper"):
        return 4
    elif(name == "paper"):
        return 5
    elif(name == "cardboard"):
        return 6
    elif(name == "cup"):
        return 7
    elif(name == "scrap"):
        return 8
    else:
        print("\n\nERROR\n\n")
        return 9

def conv_names_to_ids(names):
    output = []
    for name in names:
        output.append(conv_name_to_id(name))
    return output

def create_tf_example(example):
  # TODO(user): Populate the following variables from your example.
  height = int(example.find('size').find('height').text) # Image height
  width = int(example.find('size').find('width').text) # Image width
  filename = example.find('filename').text # Filename of the image. Empty if image is not from file
  encoded_image_data = example.find('path').text # Encoded image bytes
  image_format = b'jpeg' # b'jpeg' or b'png'

  xmins = example.iter('xmin') # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = example.iter('xmax') # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = example.iter('ymin') # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = example.iter('ymax') # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = example.iter('name') # List of string class name of bounding box (1 per box)
  classes = conv_names_to_ids(example.iter('name')) # List of integer class id of bounding box (1 per box)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


def main(_):
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  for file in os.listdir(annotation_dir):
      filename = os.fsdecode(file)
      tf_example = create_tf_example(ET.parse(annotation_path + filename).getroot())
      writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()

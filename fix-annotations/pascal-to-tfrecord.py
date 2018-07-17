import tensorflow as tf
import sys
import os
import xml.etree.ElementTree as ET

from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

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
        print(name)
        return 9

def conv_names_to_ids(names):
    output = []
    for name in names:
        output.append(conv_name_to_id(name.text))
    return output

def create_tf_example(example):
  # TODO(user): Populate the following variables from your example.
  height = int(example.find('size').find('height').text) # Image height
  width = int(example.find('size').find('width').text) # Image width
  filename = tf.compat.as_bytes(example.find('filename').text) # Filename of the image. Empty if image is not from file
  with open(example.find('path').text, "rb") as imageFile:
      f = imageFile.read()
      b = bytes(f)
      encoded_image_data = b # Encoded image bytes
  image_format = b'jpeg' # b'jpeg' or b'png'

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
  classes_text = [] # List of string class name of bounding box (1 per box)
  if(width == 0):
      print(filename)
  for val in example.iter('xmin'):
    xmins.append(int(val.text)/width)
  for val in example.iter('xmax'):
    xmaxs.append(int(val.text)/width)
  for val in example.iter('ymin'):
      ymins.append(int(val.text)/height)
  for val in example.iter('ymax'):
      ymaxs.append(int(val.text)/height)
  for val in example.iter('name'):
      classes_text.append(bytes(val.text, 'utf-8'))
  classes = conv_names_to_ids(example.iter('name')) # List of integer class id of bounding box (1 per box)

  if(9 in classes):
      print(filename)

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

  annotation_path = sys.argv[1]
  annotation_dir = os.fsencode(annotation_path)

  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  for file in os.listdir(annotation_dir):
      filename = os.fsdecode(file)
      tf_example = create_tf_example(ET.parse(annotation_path + filename).getroot())
      writer.write(tf_example.SerializeToString())

  writer.close()

  print("\nDone creating tfrecord")


if __name__ == '__main__':
  tf.app.run()

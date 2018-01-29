"""Translate an image to another image
An example of command-line usage is:
python export_graph.py --model pretrained/apple2orange.pb \
                       --input input_sample.jpg \
                       --output output_sample.jpg \
                       --image_size 256
"""

import tensorflow as tf
import os
from model import CycleGAN
import utils

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model_dir', '', '')
tf.flags.DEFINE_string('input_dir', '', '')
tf.flags.DEFINE_string('output_dir', '', '')
tf.flags.DEFINE_integer('full_image_size', 256, '')

def inference():
  graph = tf.Graph()

  try:
    os.mkdir(FLAGS.output_dir)
  except:
    pass

  with graph.as_default():
    image_data_placeholder = tf.placeholder(dtype=tf.string)
    input_image = tf.image.decode_jpeg(image_data_placeholder, channels=3) # jpeg -> uint8
    input_image = tf.image.resize_images(input_image, size=(FLAGS.full_image_size, FLAGS.full_image_size))
    input_image = utils.convert2float(input_image)
    input_image.set_shape([FLAGS.full_image_size, FLAGS.full_image_size, 3])

    for model_path in os.listdir(FLAGS.model_dir):
      print('inference model_path=%s'%model_path)

      with tf.gfile.FastGFile(FLAGS.model_dir + '/' + model_path, 'rb') as model_file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(model_file.read())

      [output_image] = tf.import_graph_def(graph_def,
                            input_map={'input_image': input_image},
                            return_elements=['output_image:0'],
                            name='output')

      with tf.Session(graph=graph) as sess:
        for input_name in os.listdir(FLAGS.input_dir):
          print('inference input_name=%s'%input_name)
          input_path = FLAGS.input_dir + '/' + input_name
          with tf.gfile.FastGFile(input_path, 'rb') as f:
            image_data = f.read()
          generated = sess.run(output_image,
              feed_dict={image_data_placeholder: image_data})
          output_path = FLAGS.output_dir + '/' + \
              os.path.splitext(input_name)[0] + '-' + os.path.splitext(model_path)[0] + '.jpg'
          with open(output_path, 'wb') as f:
            f.write(generated)

def main(unused_argv):
  inference()

if __name__ == '__main__':
  tf.app.run()

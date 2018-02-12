"""Translate an image to another image
An example of command-line usage is:
python export_graph.py --model pretrained/apple2orange.pb \
                       --input input_sample.jpg \
                       --output output_sample.jpg \
                       --image_size 256
"""

import tensorflow as tf
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
import shutil
import cv2
import glob

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model_base_dir', '', '')
tf.flags.DEFINE_string('input_dir', '', '')
tf.flags.DEFINE_string('output_dir', '', '')
tf.flags.DEFINE_string('checkpoint_arr', '', '')
tf.flags.DEFINE_string('step_arr', '', '')
tf.flags.DEFINE_integer('full_image_size', 256, '')

SPLITTOR = ','

def inference(graph, input_image, image_data_placeholder, model_dir, model_pb_name, identity=False):
  print('inference (%s, %s)'%(model_dir, model_pb_name))

  with tf.gfile.FastGFile(model_dir + '/' + model_pb_name, 'rb') as model_file:
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
      # print(input_name, model_dir, model_pb_name)
      output_full_path = '{}/{}_{}_{}.jpg'.format(FLAGS.output_dir, \
          os.path.splitext(input_name)[0], \
          model_dir[model_dir.find('/')+1:] if not identity else '00000000-0000', \
          model_pb_name[model_pb_name.find('-')+1:model_pb_name.rfind('.')])
      print('output', output_full_path)
      with open(output_full_path, 'wb') as f:
        f.write(generated)

def crop_all():
  y1, y2 = 40, 100
  x1, x2 = 30, 130
  for file_path_name in glob.glob(FLAGS.output_dir + '/*.*'):
    file_name = os.path.split(file_path_name)[1]
    print(file_path_name, file_name)
    print('crop: %s'%file_path_name)
    img = cv2.imread(file_path_name, 1)
    cropped = img[y1:y2,x1:x2]
    cv2.imwrite('{}/{}'.format(FLAGS.output_dir + '_cropped', file_name), cropped)

def main(unused_argv):
  graph = tf.Graph()

  try:
    os.mkdir(FLAGS.output_dir)
  except:
    pass
  try:
    os.mkdir(FLAGS.output_dir + '_cropped')
  except:
    pass

  with graph.as_default():
    image_data_placeholder = tf.placeholder(dtype=tf.string)
    input_image = tf.image.decode_jpeg(image_data_placeholder, channels=3) # jpeg -> uint8
    input_image = tf.image.resize_images(input_image, size=(FLAGS.full_image_size, FLAGS.full_image_size))
    input_image = utils.convert2float(input_image)
    input_image.set_shape([FLAGS.full_image_size, FLAGS.full_image_size, 3])

    inference(graph, input_image, image_data_placeholder, FLAGS.model_base_dir, 'identity.pb', identity=True)
    for checkpoint_str in FLAGS.checkpoint_arr.split(SPLITTOR):
      model_dir = FLAGS.model_base_dir + '/' + checkpoint_str
      for model_pb_name in os.listdir(model_dir):
        if model_pb_name[model_pb_name.find('-')+1:model_pb_name.rfind('.')] in FLAGS.step_arr.split(SPLITTOR):
          inference(graph, input_image, image_data_placeholder, model_dir, model_pb_name)
  
  crop_all()

if __name__ == '__main__':
  tf.app.run()


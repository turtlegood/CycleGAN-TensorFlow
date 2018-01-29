""" Freeze variables and convert 2 generator networks to 2 GraphDef files.
This makes file size smaller and can be used for inference in production.
An example of command-line usage is:
python export_graph.py --checkpoint_dir checkpoints/20170424-1152 \
                       --XtoY_model apple2orange.pb \
                       --YtoX_model orange2apple.pb \
                       --image_size 256
"""

import tensorflow as tf
import os
from tensorflow.python.tools.freeze_graph import freeze_graph
from model import CycleGAN
import utils

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('name', '', '')
tf.flags.DEFINE_integer('full_image_size', 256, '')
tf.flags.DEFINE_integer('eye_image_size', 256, '')
tf.flags.DEFINE_string('face_model_path', '', '')
tf.flags.DEFINE_integer('eye_y', '', '')
tf.flags.DEFINE_float('lambda_pix', -1, '')
tf.flags.DEFINE_integer('ngf', 64,
                        'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')

# def export_graph(model_name, XtoY=True):
def export_graph():
  graph = tf.Graph()

  with graph.as_default():
    cycle_gan = CycleGAN(
        FLAGS=FLAGS,
        ngf=FLAGS.ngf,
        norm=FLAGS.norm,
        eye_image_size=FLAGS.eye_image_size,
        full_image_size=FLAGS.full_image_size,
        face_model_path=FLAGS.face_model_path,
        eye_y=FLAGS.eye_y
    )

    input_image = tf.placeholder(tf.float32,
      shape=[FLAGS.full_image_size, FLAGS.full_image_size, 3], name='input_image')
    cycle_gan.model()
    # if XtoY:
      # output_image = cycle_gan.G.sample(tf.expand_dims(input_image, 0))
    # else:
      # output_image = cycle_gan.F.sample(tf.expand_dims(input_image, 0))
    # TODO: TOTALLY WRONG!!!
    output_image = cycle_gan.G.sample(tf.expand_dims(input_image, 0))

    output_image = tf.identity(output_image, name='output_image')
    restore_saver = tf.train.Saver()
    export_saver = tf.train.Saver()

  with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    # latest_ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    checkpoint_dir = 'checkpoints/' + FLAGS.name
    all_ckpt_list = tf.train.get_checkpoint_state(checkpoint_dir).all_model_checkpoint_paths
    for ckpt in all_ckpt_list:
      print('exporting ckpt=%s'%ckpt)
      model_dir = 'pretrained/' + FLAGS.name
      model_name = os.path.basename(ckpt) + '.pb'
      restore_saver.restore(sess, ckpt)
      output_graph_def = tf.graph_util.convert_variables_to_constants(
          sess, graph.as_graph_def(), [output_image.op.name])
      tf.train.write_graph(output_graph_def, model_dir, model_name, as_text=False)
  
def main(unused_argv):
  export_graph()
  # print('Export XtoY model...')
  # export_graph(FLAGS.XtoY_model, XtoY=True)
  # print('Does not export YtoX model')
  # print('Export YtoX model...')
  # export_graph(FLAGS.YtoX_model, XtoY=False)

if __name__ == '__main__':
  tf.app.run()

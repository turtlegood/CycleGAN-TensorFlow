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
from ast import literal_eval

tf.flags.DEFINE_string('name', '', '')
tf.flags.DEFINE_bool('export_identity', False, '')

# def export_graph(model_name, XtoY=True):
def export_graph(XtoY, export_identity):
  graph = tf.Graph()

  checkpoint_dir = 'checkpoints/' + tf.flags.FLAGS.name
  with open(checkpoint_dir + '/FLAGS.txt') as f:
    FLAGS = utils.Map(literal_eval(f.read()))

  print('loaded FLAGS: {}'.format(FLAGS))

  with graph.as_default():
    cycle_gan = CycleGAN(
        FLAGS=FLAGS,
        ngf=FLAGS.ngf,
        norm=FLAGS.norm
    )

    input_image = tf.placeholder(tf.float32,
      shape=[FLAGS.full_image_size, FLAGS.full_image_size, 3], name='input_image')

    cycle_gan.model()
    network = cycle_gan.G if XtoY else cycle_gan.F

    input_expanded = tf.expand_dims(input_image, 0)
    input_ll, input_fr = utils.full_to_eye(input_expanded, FLAGS)
    if export_identity:
      output_ll, output_fr = input_ll, input_fr
    else:
      output_ll, output_fr = cycle_gan.G(input_ll), cycle_gan.G(input_fr)
    output_full = utils.eye_to_full(input_expanded, input_ll, input_fr, output_ll, output_fr, FLAGS)
    output_int = utils.batch_convert2int(output_full)
    output_image = tf.image.encode_jpeg(tf.squeeze(output_int, [0]))

    output_image = tf.identity(output_image, name='output_image')
    restore_saver = tf.train.Saver()
    export_saver = tf.train.Saver()

  with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    # latest_ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    all_ckpt_list = tf.train.get_checkpoint_state(checkpoint_dir).all_model_checkpoint_paths
    if export_identity:
      print('exporting identity')
      model_dir = 'pretrained'
      model_name = 'identity.pb'
      output_graph_def = tf.graph_util.convert_variables_to_constants(
          sess, graph.as_graph_def(), [output_image.op.name])
      tf.train.write_graph(output_graph_def, model_dir, model_name, as_text=False)
    else:
      for ckpt in all_ckpt_list:
        print('exporting ckpt=%s'%ckpt)
        model_dir = 'pretrained/' + tf.flags.FLAGS.name
        model_name = os.path.basename(ckpt) + '.pb'
        restore_saver.restore(sess, ckpt)
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), [output_image.op.name])
        tf.train.write_graph(output_graph_def, model_dir, model_name, as_text=False)
  
def main(unused_argv):
  export_graph(XtoY=True, export_identity=tf.flags.FLAGS.export_identity)
  # print('Export XtoY model...')
  # export_graph(FLAGS.XtoY_model, XtoY=True)
  # print('Does not export YtoX model')
  # print('Export YtoX model...')
  # export_graph(FLAGS.YtoX_model, XtoY=False)

if __name__ == '__main__':
  tf.app.run()


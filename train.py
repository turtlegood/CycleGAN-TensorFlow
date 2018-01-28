import tensorflow as tf
from model import CycleGAN
from reader import Reader
from datetime import datetime
import os
import logging
from utils import ImagePool

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_string('face_model_path', '', '')
tf.flags.DEFINE_integer('full_image_size', 256, '')
tf.flags.DEFINE_integer('g_image_size', 256, '')
tf.flags.DEFINE_integer('eye_y', 128, '')
tf.flags.DEFINE_bool('use_lsgan', True,
                     'use lsgan (mean squared error) or cross entropy loss, default: True')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')
# tf.flags.DEFINE_integer('lambda1', 10.0,
#                         'weight for forward cycle loss (X->Y->X), default: 10.0')
# tf.flags.DEFINE_integer('lambda2', 10.0,
#                         'weight for backward cycle loss (Y->X->Y), default: 10.0')
tf.flags.DEFINE_float('lambda_face', 1.0, '')
tf.flags.DEFINE_float('learning_rate', 2e-4,
                      'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5,
                      'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('pool_size', 50,
                      'size of image buffer that stores previously generated images, default: 50')
tf.flags.DEFINE_integer('ngf', 64,
                        'number of gen filters in first conv layer, default: 64')

tf.flags.DEFINE_string('X', 'data/tfrecords/apple.tfrecords',
                       'X tfrecords file for training, default: data/tfrecords/apple.tfrecords')
tf.flags.DEFINE_string('Y', 'data/tfrecords/orange.tfrecords',
                       'Y tfrecords file for training, default: data/tfrecords/orange.tfrecords')
tf.flags.DEFINE_string('load_model', None,
                        'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
tf.flags.DEFINE_string('train_name', None,
                        'the custom name of the training which will be shown in the folder name, default: None')


def train():
  if FLAGS.load_model is not None:
    checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
  else:
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    postfix = ('-' + FLAGS.train_name) if FLAGS.train_name is not None else ''
    checkpoints_dir = "checkpoints/{}{}".format(current_time, postfix)
    try:
      os.makedirs(checkpoints_dir)
    except os.error:
      pass
  
  print('checkpoints_dir: {}'.format(checkpoints_dir))
  print('''suggested tensorboard command:

  tensorboard --logdir ./{}

  '''.format(checkpoints_dir))

  graph = tf.Graph()
  with graph.as_default():
    cycle_gan = CycleGAN(
        X_train_file=FLAGS.X,
        Y_train_file=FLAGS.Y,
        face_model_path=FLAGS.face_model_path,
        batch_size=FLAGS.batch_size,
        full_image_size=FLAGS.full_image_size,
        eye_y=FLAGS.eye_y,
        g_image_size=FLAGS.g_image_size,
        use_lsgan=FLAGS.use_lsgan,
        norm=FLAGS.norm,
        lambda_face=FLAGS.lambda_face,
        learning_rate=FLAGS.learning_rate,
        beta1=FLAGS.beta1,
        ngf=FLAGS.ngf
    )

    # XXX
    # G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x = cycle_gan.model()
    # optimizers = cycle_gan.optimize(G_loss, D_Y_loss, F_loss, D_X_loss)
    G_loss, D_Y_loss, face_loss, fake_y = cycle_gan.model()
    optimizers = cycle_gan.optimize(G_loss, D_Y_loss, face_loss)

    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
    saver = tf.train.Saver()

  with tf.Session(graph=graph) as sess:
    if FLAGS.load_model is not None:
      checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
      meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
      restore = tf.train.import_meta_graph(meta_graph_path)
      restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
      step = int(meta_graph_path.split("-")[2].split(".")[0])
    else:
      sess.run(tf.global_variables_initializer())
      step = 0

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
      fake_Y_pool = ImagePool(FLAGS.pool_size)
      # XXX
      # fake_X_pool = ImagePool(FLAGS.pool_size)

      while not coord.should_stop():
        # get previously generated images
        # XXX
        # fake_y_val, fake_x_val = sess.run([fake_y, fake_x])
        # NOT: fake_y_val = sess.run([fake_y])
        fake_y_val = sess.run(fake_y)

        # train
        # _, G_loss_val, D_Y_loss_val, F_loss_val, D_X_loss_val, summary = (
        #       sess.run(
        #           [optimizers, G_loss, D_Y_loss, F_loss, D_X_loss, summary_op],
        #           feed_dict={cycle_gan.fake_y: fake_Y_pool.query(fake_y_val),
        #                      cycle_gan.fake_x: fake_X_pool.query(fake_x_val)}
        #       )
        # )

        _, G_loss_val, D_Y_loss_val, face_loss_val, summary = (
              sess.run(
                  [optimizers, G_loss, D_Y_loss, face_loss, summary_op],
                  feed_dict={cycle_gan.fake_y: fake_Y_pool.query(fake_y_val)}
              )
        )

        if step < 500 or step % 10 == 0:
          train_writer.add_summary(summary, step)
          train_writer.flush()

        if step % 100 == 0:
          logging.info('-----------Step %d:-------------' % step)
          logging.info('  G_loss    : {}'.format(G_loss_val))
          logging.info('  D_Y_loss  : {}'.format(D_Y_loss_val))
          logging.info('  face_loss : {}'.format(face_loss_val))
          # XXX
          # logging.info('  F_loss   : {}'.format(F_loss_val))
          # logging.info('  D_X_loss : {}'.format(D_X_loss_val))

        if step % 10000 == 0:
          save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
          logging.info("Model saved in file: %s" % save_path)

        step += 1

    except KeyboardInterrupt:
      logging.info('Interrupted')
      coord.request_stop()
    except Exception as e:
      coord.request_stop(e)
    finally:
      save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
      logging.info("Model saved in file: %s" % save_path)
      # When done, ask the threads to stop.
      coord.request_stop()
      coord.join(threads)

def main(unused_argv):
  train()

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  tf.app.run()

import tensorflow as tf
from model import CycleGAN
from reader import Reader
from datetime import datetime
import os
import logging
from utils import ImagePool

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_bool('use_G_new_tanh', False, '')
tf.flags.DEFINE_bool('use_wgan_gp', False, '')
tf.flags.DEFINE_integer('num_critic_train', 5, '')
tf.flags.DEFINE_float('lambda_gp', 10.0, '')

tf.flags.DEFINE_string('face_model_path', '', '')
tf.flags.DEFINE_integer('full_image_size', 0, '')
tf.flags.DEFINE_integer('eye_image_size', 0, '')
tf.flags.DEFINE_integer('eye_y', 0, '')

tf.flags.DEFINE_bool('use_faceloss_prewhitten', False, '')
tf.flags.DEFINE_bool('use_G_skip_conn', False, '')
tf.flags.DEFINE_bool('use_G_resi', False, '')

tf.flags.DEFINE_bool('formal', False, '')
tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_bool('use_lsgan', True, 'use lsgan (mean squared error) or cross entropy loss, default: True')

tf.flags.DEFINE_integer('lambda1', 10.0, 'weight for forward cycle loss (X->Y->X), default: 10.0')
tf.flags.DEFINE_integer('lambda2', 10.0, 'weight for backward cycle loss (Y->X->Y), default: 10.0')
tf.flags.DEFINE_float('lambda_face', 1.0, '')
tf.flags.DEFINE_float('lambda_pix', 1.0, '')
tf.flags.DEFINE_float('lambda_gan', 1.0, '')

tf.flags.DEFINE_float('lr_G', -1, '')
tf.flags.DEFINE_float('lr_D', -1, '')

tf.flags.DEFINE_integer('ngf', 64, 'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_string('norm', 'instance', '[instance, batch] use instance norm or batch norm, default: instance')
tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('pool_size', 50, 'size of image buffer that stores previously generated images, default: 50')

tf.flags.DEFINE_string('X', 'data/tfrecords/apple.tfrecords', 'X tfrecords file for training, default: data/tfrecords/apple.tfrecords')
tf.flags.DEFINE_string('Y', 'data/tfrecords/orange.tfrecords', 'Y tfrecords file for training, default: data/tfrecords/orange.tfrecords')
tf.flags.DEFINE_string('load_model', None, 'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
tf.flags.DEFINE_string('train_name', None, 'the custom name of the training which will be shown in the folder name, default: None')

def train():
  checkpoints_dir_prefix = 'checkpoints/' if FLAGS.formal else 'checkpoints_informal/'
  if FLAGS.load_model is not None:
    # checkpoints_dir = checkpoints_dir_prefix + FLAGS.load_model.lstrip(checkpoints_dir_prefix)
    checkpoints_dir = checkpoints_dir_prefix + FLAGS.load_model
  else:
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    postfix = ('-' + FLAGS.train_name) if FLAGS.train_name is not None else ''
    checkpoints_dir = "{}{}{}".format(checkpoints_dir_prefix, current_time, postfix)
    try:
      os.makedirs(checkpoints_dir)
    except os.error:
      pass
  
  print('checkpoints_dir: {}'.format(checkpoints_dir))
  print('''suggested tensorboard command:

  tensorboard --logdir ~/TomChen/Sync/CycleGAN-TensorFlow/{}

  '''.format(checkpoints_dir))

  # save cfg
  with open(checkpoints_dir+'/FLAGS.txt', 'w') as f:
    f.write(str(FLAGS.__flags))

  graph = tf.Graph()
  with graph.as_default():
    cycle_gan = CycleGAN(
        FLAGS=FLAGS,
        X_train_file=FLAGS.X,
        Y_train_file=FLAGS.Y,
        batch_size=FLAGS.batch_size,
        use_lsgan=FLAGS.use_lsgan,
        norm=FLAGS.norm,
        lambda1=FLAGS.lambda1,
        lambda2=FLAGS.lambda2,
        beta1=FLAGS.beta1,
        ngf=FLAGS.ngf
    )
    (G_loss, D_Y_loss, F_loss, D_X_loss), (fake_y, fake_x) = cycle_gan.model()
    (generator_optimizers, discriminator_optimizers) = cycle_gan.optimize(G_loss, D_Y_loss, F_loss, D_X_loss)

    summary_op = tf.summary.merge_all()
    secondary_summary_op = tf.summary.merge_all(key='summaries_secondary')
    train_writer = tf.summary.FileWriter(checkpoints_dir + '/main', graph)
    train_secondary_writer = tf.summary.FileWriter(checkpoints_dir + '/secondary', graph)
    saver = tf.train.Saver(max_to_keep=100) # XXX

  with tf.Session(graph=graph) as sess:
    if FLAGS.load_model is not None:
      checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
      meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
      restore = tf.train.import_meta_graph(meta_graph_path)
      restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
      step = int(meta_graph_path.split("-")[-1].split(".")[0])
    else:
      sess.run(tf.global_variables_initializer())
      step = 0

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
      fake_Y_pool = ImagePool(FLAGS.pool_size)
      fake_X_pool = ImagePool(FLAGS.pool_size)

      while not coord.should_stop():

        # train D
        for _ in range(FLAGS.num_critic_train):
          # get previously generated images
          fake_y_val, fake_x_val = sess.run([fake_y, fake_x])
          _ = sess.run([discriminator_optimizers], \
                feed_dict={cycle_gan.fake_y_full: fake_Y_pool.query(fake_y_val), \
                            cycle_gan.fake_x_full: fake_X_pool.query(fake_x_val)}
          )
        
        # TODO: batch_size ?
                            
        # train G
        # TODO: ??? that feed_dict
        _, summary, secondary_summary = ( \
              sess.run( \
                  [generator_optimizers, summary_op, secondary_summary_op], \
                  feed_dict={cycle_gan.fake_y_full: fake_Y_pool.query(fake_y_val), \
                            cycle_gan.fake_x_full: fake_X_pool.query(fake_x_val)}
              ) \
        )

        if step < 500 or step % 10 == 0:
          train_writer.add_summary(summary, step)
          train_writer.flush()
          train_secondary_writer.add_summary(secondary_summary, step)
          train_secondary_writer.flush()

        if step % 50 == 0:
          logging.info('Step: %d' % step)

        if step % 10000 == 0:
          if FLAGS.formal:
            save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
            logging.info("Model saved in file: %s" % save_path)
          else:
            logging.info("Model not saved since it is informal")

        step += 1

    except KeyboardInterrupt:
      logging.info('Interrupted')
      coord.request_stop()
    except Exception as e:
      coord.request_stop(e)
    finally:
      if FLAGS.formal:
        save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
        logging.info("Model saved in file: %s" % save_path)
      else:
        logging.info("Model not saved since it is informal")
      # When done, ask the threads to stop.
      coord.request_stop()
      coord.join(threads)

def main(unused_argv):
  train()

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  tf.app.run()

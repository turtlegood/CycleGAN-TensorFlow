import tensorflow as tf
import ops
import utils
from reader import Reader
from discriminator import Discriminator
from generator import Generator
import facenet_loss

REAL_LABEL = 0.9

class CycleGAN:
  def __init__(self,
               FLAGS=None,
               X_train_file='',
               Y_train_file='',
               batch_size=1,
               use_lsgan=True,
               norm='instance',
               lambda1=10.0,
               lambda2=10.0,
               beta1=0.5,
               ngf=64
              ):
    """
    Args:
      X_train_file: string, X tfrecords file for training
      Y_train_file: string Y tfrecords file for training
      batch_size: integer, batch size
      image_size: integer, image size
      lambda1: integer, weight for forward cycle loss (X->Y->X)
      lambda2: integer, weight for backward cycle loss (Y->X->Y)
      use_lsgan: boolean
      norm: 'instance' or 'batch'
      learning_rate: float, initial learning rate for Adam
      beta1: float, momentum term of Adam
      ngf: number of gen filters in first conv layer
    """
    self.FLAGS = FLAGS
    self.lambda1 = lambda1
    self.lambda2 = lambda2
    self.use_lsgan = use_lsgan
    use_sigmoid = not use_lsgan
    self.batch_size = batch_size
    self.beta1 = beta1
    self.X_train_file = X_train_file
    self.Y_train_file = Y_train_file

    self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')

    self.G = Generator('G', self.is_training,
        ngf=ngf, norm=norm, image_size=self.FLAGS.eye_image_size, FLAGS=self.FLAGS)
    self.D_Y = Discriminator('D_Y',
        self.is_training, norm=norm, use_sigmoid=use_sigmoid)
    self.F = Generator('F', self.is_training,
        norm=norm, image_size=self.FLAGS.eye_image_size, FLAGS=self.FLAGS)
    self.D_X = Discriminator('D_X',
        self.is_training, norm=norm, use_sigmoid=use_sigmoid)

    self.fake_x_full = tf.placeholder(tf.float32,
        shape=[batch_size, self.FLAGS.full_image_size, self.FLAGS.full_image_size, 3])
    self.fake_y_full = tf.placeholder(tf.float32,
        shape=[batch_size, self.FLAGS.full_image_size, self.FLAGS.full_image_size, 3])

  def model(self):
    ### image reading ###

    X_reader = Reader(self.X_train_file, name='X',
        image_size=self.FLAGS.full_image_size, batch_size=self.batch_size)
    Y_reader = Reader(self.Y_train_file, name='Y',
        image_size=self.FLAGS.full_image_size, batch_size=self.batch_size)

    x_full = X_reader.feed()
    y_full = Y_reader.feed()

    ### image processing ###

    x_concated = utils.full_to_eye_concated(x_full, self.FLAGS)
    y_concated = utils.full_to_eye_concated(y_full, self.FLAGS)
    self_fake_x_concated = utils.full_to_eye_concated(self.fake_x_full, self.FLAGS)
    self_fake_y_concated = utils.full_to_eye_concated(self.fake_y_full, self.FLAGS)

    fake_y_concated = self.G(x_concated)
    fake_x_concated = self.F(y_concated)
    cycle_x_concated = self.F(fake_y_concated)
    cycle_y_concated = self.G(fake_x_concated)

    fake_y_full = utils.eye_concated_to_full(x_full, x_concated, fake_y_concated, self.FLAGS)    
    fake_x_full = utils.eye_concated_to_full(y_full, y_concated, fake_x_concated, self.FLAGS)    

    ### losses ###

    cycle_loss = self.cycle_consistency_loss(x_concated, y_concated, cycle_x_concated, cycle_y_concated)

    #TODO: some are average, some are sum - !
    #TODO: instance norm ?

    # X -> Y
    G_gan_loss = self.generator_loss(self.D_Y, fake_y_concated)
    D_Y_loss = self.discriminator_loss(self.D_Y, y_concated, self_fake_y_concated, 'D_Y')

    # Y -> X
    F_gan_loss = self.generator_loss(self.D_X, fake_x_concated)
    D_X_loss = self.discriminator_loss(self.D_X, x_concated, self_fake_x_concated, 'D_X')

    # face_loss
    if self.FLAGS.lambda_face == 0:
      G_face_loss = F_face_loss = 0
    else:
      G_face_loss, F_face_loss = facenet_loss.facenet_loss(
          tf.concat([x_full, fake_y_full, y_full, fake_x_full], 0), concat_size=2, FLAGS=self.FLAGS)
    
    # pix loss
    if self.FLAGS.lambda_pix == 0:
      G_pix_loss = F_pix_loss = 0
    else:
      G_pix_loss = self.FLAGS.lambda_pix * tf.norm(fake_y_full - x_full, 1)
      F_pix_loss = self.FLAGS.lambda_pix * tf.norm(fake_x_full - y_full, 1)
    
    # total loss
    # XXX WRONG AND WRONG! all of the lambdas are multiplied TWICE!
    # G_loss = G_gan_loss + cycle_loss + self.FLAGS.lambda_face * G_face_loss + self.FLAGS.lambda_pix * G_pix_loss
    # F_loss = F_gan_loss + cycle_loss + self.FLAGS.lambda_face * F_face_loss + self.FLAGS.lambda_pix * F_pix_loss
    G_loss = G_gan_loss + cycle_loss + G_face_loss + G_pix_loss
    F_loss = F_gan_loss + cycle_loss + F_face_loss + F_pix_loss

    ### summaries ###

    # utils.summary_batch(names=['fake_x_full', 'fake_y_full'], locals=locals(), prefix='dbg')
    # utils.summary_float_image('dbg/delta_fake_y_and_x', fake_y_full - x_full)

    utils.summary_scalar('loss/G_gan', G_gan_loss)
    utils.summary_scalar('loss/D_Y', D_Y_loss)
    utils.summary_scalar('loss/F_gan', F_gan_loss)
    utils.summary_scalar('loss/D_X', D_X_loss)
    utils.summary_scalar('loss/cycle', cycle_loss)
    utils.summary_scalar('loss/G_face', G_face_loss)
    utils.summary_scalar('loss/F_face', F_face_loss)
    utils.summary_scalar('loss/G_pix', G_pix_loss)
    utils.summary_scalar('loss/F_pix', F_pix_loss)
    utils.summary_scalar('loss/G_sum', G_loss)
    utils.summary_scalar('loss/F_sum', F_loss)

    # tf.summary.histogram('D_Y/true', self.D_Y(y))
    # tf.summary.histogram('D_Y/fake', self.D_Y(self.G(x)))
    # tf.summary.histogram('D_X/true', self.D_X(x))
    # tf.summary.histogram('D_X/fake', self.D_X(self.F(y)))

    utils.summary_float_image('X/1_input', x_full)
    utils.summary_float_image('Y/1_input', y_full)
    utils.summary_float_image('X/2_generated', fake_y_full)
    utils.summary_float_image('Y/2_generated', fake_x_full)
    # utils.summary_float_image('X_ll/3reconstruction', self.F(self.G(x_ll)))
    # utils.summary_float_image('Y_ll/3reconstruction', self.G(self.F(y_ll)))

    utils.summary_float_image('X_concated/1_input', x_concated)
    utils.summary_float_image('Y_concated/1_input', y_concated)
    utils.summary_float_image('X_concated/2_generated', fake_y_concated)
    utils.summary_float_image('Y_concated/2_generated', fake_x_concated)
    utils.summary_float_image('X_concated/3_reconstruction', cycle_x_concated)
    utils.summary_float_image('Y_concated/3_reconstruction', cycle_y_concated)
    utils.summary_float_image('X_concated/4_residual', fake_y_concated - x_concated)
    utils.summary_float_image('Y_concated/4_residual', fake_x_concated - y_concated)

    return  (G_loss, D_Y_loss, F_loss, D_X_loss), (fake_y_full, fake_x_full)
  
  def optimize(self, G_loss, D_Y_loss, F_loss, D_X_loss):
    def make_optimizer(loss, variables, lr, name='Adam'):
      if loss is None:
        print('Ignore loss {} because it is None'.format(name))
      else:
        """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
            and a linearly decaying rate that goes to zero over the next 100k steps
        """
        global_step = tf.Variable(0, trainable=False)
        # starter_learning_rate = lr
        # end_learning_rate = 0.0
        # start_decay_step = 100000
        # decay_steps = 100000
        beta1 = self.beta1
        # learning_rate = (
        #     tf.where(
        #             tf.greater_equal(global_step, start_decay_step),
        #             tf.train.polynomial_decay(starter_learning_rate, global_step-start_decay_step,
        #                                       decay_steps, end_learning_rate,
        #                                       power=1.0),
        #             starter_learning_rate
        #     )
        # )
        # tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)
        # learning_step = (
        #     tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
        #             .minimize(loss, global_step=global_step, var_list=variables)
        # )
        learning_step = (
            tf.train.AdamOptimizer(lr, beta1=beta1, name=name)
                    .minimize(loss, global_step=global_step, var_list=variables)
        )
        return learning_step

    G_optimizer = make_optimizer(G_loss, self.G.variables, self.FLAGS.lr_G, name='Adam_G')
    D_Y_optimizer = make_optimizer(D_Y_loss, self.D_Y.variables, self.FLAGS.lr_D, name='Adam_D_Y')
    F_optimizer =  make_optimizer(F_loss, self.F.variables, self.FLAGS.lr_G, name='Adam_F')
    D_X_optimizer = make_optimizer(D_X_loss, self.D_X.variables, self.FLAGS.lr_D, name='Adam_D_X')

    with tf.control_dependencies([G_optimizer, F_optimizer]):
      generator_optimizers = tf.no_op(name='optimizers')
    with tf.control_dependencies([D_Y_optimizer, D_X_optimizer]):
      discriminator_optimizers = tf.no_op(name='optimizers')
    return (generator_optimizers, discriminator_optimizers)

  def discriminator_loss(self, D, y, fake_y, name):
    """ Note: default: D(y).shape == (batch_size,5,5,1),
                       fake_buffer_size=50, batch_size=1
    Args:
      G: generator object
      D: discriminator object
      y: 4D tensor (batch_size, image_size, image_size, 3)
    Returns:
      loss: scalar
    """
    if self.FLAGS.use_wgan_gp:
      error_real = -1.0 * tf.reduce_mean(D(y))
      error_fake = tf.reduce_mean(D(fake_y))
      loss_classical = error_real + error_fake

      # 2* because it has TWO eyes
      alpha = tf.random_uniform(shape=[2 * self.FLAGS.batch_size, 1, 1, 1], minval=0.,maxval=1.)
      y_hat = alpha * y + (1.0-alpha) * fake_y

      # TODO: this [0] ?
      # TODO: [1,2,3] ?
      # TODO: debug this 
      loss_gp = self.FLAGS.lambda_gp * tf.reduce_mean(
              ( tf.sqrt(tf.reduce_sum(
                    tf.gradients(D(y_hat), y_hat)[0] **2, reduction_indices=[1,2,3]
                ))
              - 1.0) **2
            )

      utils.summary_scalar('loss/' + name + '_classical', loss_classical)
      utils.summary_scalar('loss/' + name + '_gp', loss_gp)

      loss = loss_classical + loss_gp
    elif self.use_lsgan:
      # use mean squared error
      error_real = tf.reduce_mean(tf.squared_difference(D(y), REAL_LABEL))
      error_fake = tf.reduce_mean(tf.square(D(fake_y)))
      loss = (error_real + error_fake) / 2
    else:
      # use cross entropy
      error_real = -tf.reduce_mean(ops.safe_log(D(y)))
      error_fake = -tf.reduce_mean(ops.safe_log(1-D(fake_y)))
      loss = (error_real + error_fake) / 2
    return loss

  def generator_loss(self, D, fake_y):
    """  fool discriminator into believing that G(x) is real
    """
    if self.FLAGS.use_wgan_gp:
      loss = -1.0 * tf.reduce_mean(D(fake_y))
    elif self.use_lsgan:
      # use mean squared error
      loss = tf.reduce_mean(tf.squared_difference(D(fake_y), REAL_LABEL))
    else:
      # heuristic, non-saturating loss
      loss = -tf.reduce_mean(ops.safe_log(D(fake_y))) / 2
    loss = self.FLAGS.lambda_gan * loss
    return loss

  def cycle_consistency_loss(self, x, y, cycle_x, cycle_y):
    """ cycle consistency loss (L1 norm)
    """
    forward_loss = tf.reduce_mean(tf.abs(cycle_x-x))
    backward_loss = tf.reduce_mean(tf.abs(cycle_y-y))
    # forward_loss = tf.reduce_mean(tf.abs(F(G(x))-x))
    # backward_loss = tf.reduce_mean(tf.abs(G(F(y))-y))
    loss = self.lambda1*forward_loss + self.lambda2*backward_loss
    return loss

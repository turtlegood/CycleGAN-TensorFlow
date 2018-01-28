import tensorflow as tf
import ops
import utils

class Generator:
  def __init__(self, name, is_training, ngf=64, norm='instance', full_image_size=128, g_image_size=128):
    self.name = name
    self.reuse = False
    self.ngf = ngf
    self.norm = norm
    self.is_training = is_training
    self.full_image_size = full_image_size
    self.g_image_size = g_image_size

  def __call__(self, input):
    # input: batch_size x full_image_size x full_image_size x 3
    tf.summary.image('xxx1', input)
    y_start, y_end = 70-24, 70+24
    x_start, x_end = 80-48, 80
    cropped = input[:, y_start:y_end, x_start:x_end, :]
    tf.summary.image('xxx2', cropped)
    return self.raw_call(cropped)

  # XXX originally it is this __call__
  def raw_call(self, input):
    """
    Args:
      input: batch_size x width x height x 3
    Returns:
      output: same size as input
    """
    with tf.variable_scope(self.name):
      # conv layers
      c7s1_32 = ops.c7s1_k(input, self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='c7s1_32')                             # (?, w, h, 32)
      d64 = ops.dk(c7s1_32, 2*self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='d64')                                 # (?, w/2, h/2, 64)
      d128 = ops.dk(d64, 4*self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='d128')                                # (?, w/4, h/4, 128)

      # XXX
      if self.g_image_size <= 128:
        # use 6 residual blocks for 128x128 images
        res_output = ops.n_res_blocks(d128, reuse=self.reuse, n=6)      # (?, w/4, h/4, 128)
      else:
        # 9 blocks for higher resolution
        res_output = ops.n_res_blocks(d128, reuse=self.reuse, n=9)      # (?, w/4, h/4, 128)

      # fractional-strided convolution
      u64 = ops.uk(res_output, 2*self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='u64')                                 # (?, w/2, h/2, 64)
      u32 = ops.uk(u64, self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='u32', output_size=self.g_image_size)         # (?, w, h, 32)

      # conv layer
      # Note: the paper said that ReLU and _norm were used
      # but actually tanh was used and no _norm here
      # XXX: Try to do a residual thing :/
      output = tf.nn.tanh(input + ops.c7s1_k(u32, 3, norm=None,
          activation=None, reuse=self.reuse, name='output'))           # (?, w, h, 3)
      # output = ops.c7s1_k(u32, 3, norm=None,
      #     activation='tanh', reuse=self.reuse, name='output')           # (?, w, h, 3)
    # set reuse=True for next call
    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    return output

  def sample(self, input):
    image = utils.batch_convert2int(self.__call__(input))
    image = tf.image.encode_jpeg(tf.squeeze(image, [0]))
    return image

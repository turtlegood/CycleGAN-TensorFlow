import tensorflow as tf
import random

def __eye_crop_box(eye_mode, FLAGS):
  half_full_size = FLAGS.full_image_size // 2
  half_g_size = FLAGS.eye_image_size // 2
  y_start, y_end = FLAGS.eye_y - half_g_size, FLAGS.eye_y + half_g_size
  if eye_mode == 'left':
    x_start, x_end = half_full_size - FLAGS.eye_image_size, half_full_size
  elif eye_mode == 'right':
    x_start, x_end = half_full_size, half_full_size + FLAGS.eye_image_size
  else:
    raise Exception('wrong eye_mode')
  return y_start, y_end, x_start, x_end 

def full_to_eye(full_image, FLAGS):
  def to_one_eye(input, eye_mode):
    with tf.name_scope('eye_' + eye_mode):
      y_start, y_end, x_start, x_end = __eye_crop_box(eye_mode, FLAGS)
      # crop
      cropped = input[:, y_start:y_end, x_start:x_end, :]
      # maybe flip
      if eye_mode == 'right':
        cropped = tf.reverse(cropped, [2])
      return cropped

  return to_one_eye(full_image, 'left'), to_one_eye(full_image, 'right')

def eye_to_full(ll, fr, FLAGS):
  def one_eye_to(input, eye_mode):
    with tf.name_scope('eye_rev_' + eye_mode):
      y_start, y_end, x_start, x_end = __eye_crop_box(eye_mode, FLAGS)
      if eye_mode == 'right':
        input = tf.reverse(input, [2])
      # pad back
      padded = tf.pad(input,
          [[0,0], [y_start, FLAGS.full_image_size - y_end], [x_start, FLAGS.full_image_size - x_end], [0,0]], 'CONSTANT')
      return padded

  return one_eye_to(ll, 'left') + one_eye_to(fr, 'right')


def summary_float_image(name, image, summary_histogram=True, summary_image=True):
  if summary_image:
    tf.summary.image(name, batch_convert2int(image), max_outputs=1)
  if summary_histogram:
    tf.summary.histogram(name, image)

def convert2int(image):
  """ Transfrom from float tensor ([-1.,1.]) to int image ([0,255])
  """
  return tf.image.convert_image_dtype((image+1.0)/2.0, tf.uint8)

def convert2float(image):
  """ Transfrom from int image ([0,255]) to float tensor ([-1.,1.])
  """
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  return (image/127.5) - 1.0

def batch_convert2int(images):
  """
  Args:
    images: 4D float tensor (batch_size, image_size, image_size, depth)
  Returns:
    4D int tensor
  """
  return tf.map_fn(convert2int, images, dtype=tf.uint8)

def batch_convert2float(images):
  """
  Args:
    images: 4D int tensor (batch_size, image_size, image_size, depth)
  Returns:
    4D float tensor
  """
  return tf.map_fn(convert2float, images, dtype=tf.float32)

class ImagePool:
  """ History of generated images
      Same logic as https://github.com/junyanz/CycleGAN/blob/master/util/image_pool.lua
  """
  def __init__(self, pool_size):
    self.pool_size = pool_size
    self.images = []

  def query(self, image):
    if self.pool_size == 0:
      return image

    if len(self.images) < self.pool_size:
      self.images.append(image)
      return image
    else:
      p = random.random()
      if p > 0.5:
        # use old image
        random_id = random.randrange(0, self.pool_size)
        tmp = self.images[random_id].copy()
        self.images[random_id] = image.copy()
        return tmp
      else:
        return image


from tensorflow.python.platform import gfile
from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import facenet
import math
import utils
 
# maybe only can be called once!
__facenet_loss_called = False
def facenet_loss(tensor_concated, concat_size, FLAGS, name='face_loss'):
    global __facenet_loss_called; assert __facenet_loss_called == False; __facenet_loss_called = True
    print('facenet_loss lambda_face=%f, face_model_path=%s, full_image_size=%d'%(FLAGS.lambda_face, FLAGS.face_model_path, FLAGS.full_image_size))
    with tf.name_scope(name):
        input_map = {"input:0": tensor_concated, "phase_train:0": tf.constant(False)}

        # facenet.load_model(args.model)
        model_exp = os.path.expanduser(FLAGS.face_model_path)
        # print('facenet model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            # tf.import_graph_def(graph_def, name='')
            tf.import_graph_def(graph_def, name='', input_map=input_map)
        
        # Get input and output tensors
        # XXX opt G
        # images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        # phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        # print(tf.shape(embeddings), embeddings.get_shape())

        reduced_norm_tuple = ()

        # print(tensor_concated.get_shape(), embeddings.get_shape())
        splitted_by_concat = tf.split(embeddings, num_or_size_splits=concat_size)
        for splitted in splitted_by_concat:
            x_i_emb, fake_y_i_emb = tf.split(splitted, num_or_size_splits=2)
            delta_i_emb = fake_y_i_emb - x_i_emb
            norm = tf.norm(delta_i_emb, 2, axis=1)
            multiplied_norm = FLAGS.lambda_face * norm
            reduced_norm = tf.reduce_mean(multiplied_norm)
            reduced_norm_tuple += (reduced_norm,)
            # print(splitted.get_shape(), x_i_emb.get_shape(), fake_y_i_emb.get_shape())

        return reduced_norm_tuple

# def prewhitten(input, full_image_size):
#     mean, var = tf.nn.moments(input, [0, 1, 2])
#     std = tf.sqrt(var)
#     # TODO more flexible, not requiring full_image_size
#     std_adj = tf.maximum(std,
#             1.0/math.sqrt(full_image_size * full_image_size * 3))
#     y = tf.multiply(tf.subtract(input, mean), 1/std_adj)
#     # mean = np.mean(x)
#     # std = np.std(x)
#     # std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
#     # y = np.multiply(np.subtract(x, mean), 1/std_adj)
#     return y  
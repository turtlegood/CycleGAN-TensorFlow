from tensorflow.python.platform import gfile
from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import facenet
import math
 
def facenet_loss(x, fake_y, face_model_path='', full_image_size=128):
    with tf.name_scope('facenet_loss'):
        # really good input! https://github.com/tensorflow/tensorflow/issues/1758
        # print(x[0], fake_y[0])
        whiten_x_0 = prewhitten(x[0], full_image_size)
        whiten_y_0 = prewhitten(fake_y[0], full_image_size)
        good_input = tf.stack([whiten_x_0, whiten_y_0], 0)

        input_map = {"input:0": good_input, "phase_train:0": tf.constant(False)}

        # facenet.load_model(args.model)
        model_exp = os.path.expanduser(face_model_path)
        print('facenet model filename: %s' % model_exp)
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

        x_emb = embeddings[0]
        fake_y_emb = embeddings[1]
        delta_emb = fake_y_emb - x_emb
        norm = tf.norm(delta_emb, 2)

        # tf.summary.image('no_pre', [x[0]])
        # tf.summary.image('pre', [whiten_x_0])
        # tf.summary.histogram('no_pre', [x[0]])
        # tf.summary.histogram('pre', [whiten_x_0])

        return norm

def prewhitten(input, full_image_size):
    mean, var = tf.nn.moments(input, [0, 1, 2])
    std = tf.sqrt(var)
    # TODO more flexible, not requiring full_image_size
    std_adj = tf.maximum(std,
            1.0/math.sqrt(full_image_size * full_image_size * 3))
    y = tf.multiply(tf.subtract(input, mean), 1/std_adj)
    # mean = np.mean(x)
    # std = np.std(x)
    # std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    # y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  
from tensorflow.python.platform import gfile
from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import facenet

def facenet_loss(x, fake_y, face_model_path='', full_image_size=128):
    # really good input! https://github.com/tensorflow/tensorflow/issues/1758
    # print(x[0], fake_y[0])
    good_input = tf.stack([x[0], fake_y[0]], 0)

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

    return norm
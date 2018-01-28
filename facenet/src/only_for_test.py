from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import gfile
from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import facenet
import align.detect_face

def main(args):

    # XXX
    import os
    import os.path
    args.image_files = [os.path.join(p, f) for p in args.image_files for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))]
    args.image_files.sort()
    print(args.image_files)

    images = load_and_align_data(args.image_files, args.image_size, args.margin, args.gpu_memory_fraction)
    with tf.Graph().as_default():
        with tf.Session() as sess:
      
            # Load the model
            # facenet.load_model(args.model)
            # copied
            good_input = tf.placeholder(tf.float32, shape=(2,160,160,3))
            input_map = {"input:0": good_input, "phase_train:0": tf.constant(False)}

            model_exp = os.path.expanduser(args.model)
            print('Model filename: %s' % model_exp)
            with gfile.FastGFile(model_exp,'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                # tf.import_graph_def(graph_def, name='')
                tf.import_graph_def(graph_def, name='', input_map=input_map)
            
            # Get input and output tensors
            # images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            # phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            emb0 = embeddings[0]
            emb1 = embeddings[1]
            delta = emb1 - emb0
            norm = tf.norm(delta, 2)
            
            # print(images.shape, images_placeholder, images_placeholder.get_shape(), tf.shape(images_placeholder))

            # Run forward pass to calculate embeddings
            # feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            feed_dict = { good_input: images }
            norm_val = sess.run(norm, feed_dict=feed_dict)
            print('NORM!', norm_val)

            emb = sess.run(embeddings, feed_dict=feed_dict)

            nrof_images = len(args.image_files)

            # print(tf.shape(embeddings), embeddings.get_shape())
            
            # print('Images:')
            # for i in range(nrof_images):
            #     print('%1d: %s' % (i, args.image_files[i]))
            # print('')
            
            # # Print distance matrix
            # print('Distance matrix')
            # print('    ', end='')
            # for i in range(nrof_images):
            #     print('    %1d     ' % i, end='')
            #     # print('  %1d   ' % i, end='')
            # print('')
            # for i in range(nrof_images):
            #     print('%1d  ' % i, end='')
            #     for j in range(nrof_images):
            #         dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
            #         print('  %1.4f  ' % dist, end='')
            #         # print('  %1d  ' % (dist*10), end='')
            #     print('')
            
            # train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
            #                           sess.graph)
            
            
def load_img(image_path):
    img = misc.imread(os.path.expanduser(image_path), mode='RGB')
    aligned = misc.imresize(img, (image_size, image_size), interp='bilinear')
    prewhitened = facenet.prewhiten(aligned)
    return prewhitened


def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    # minsize = 20 # minimum size of face
    # threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    # factor = 0.709 # scale factor
    
    # print('Creating networks and loading parameters')
    # with tf.Graph().as_default():
    #     # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    #     # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    #     sess = tf.Session()
    #     with sess.as_default():
    #         pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  
    tmp_image_paths = image_paths.copy()
    # import os
    # import os.path
    # image_paths = [os.path.join(p, f) for p in input_image_paths for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))]
    # tmp_image_paths = image_paths[:]
    # print(image_paths)
    
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        # img_size = np.asarray(img.shape)[0:2]
        # bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        # if len(bounding_boxes) < 1:
        #   image_paths.remove(image)
        #   print("can't detect face, remove ", image)
        #   continue
        # det = np.squeeze(bounding_boxes[0,0:4])
        # bb = np.zeros(4, dtype=np.int32)
        # bb[0] = np.maximum(det[0]-margin/2, 0)
        # bb[1] = np.maximum(det[1]-margin/2, 0)
        # bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        # bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        # cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        # aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        # prewhitened = facenet.prewhiten(aligned)
        # img_list.append(prewhitened)

        #XXX
        aligned = misc.imresize(img, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(aligned)
        # img_list.append(prewhitened)
        # print("TYPE", type(prewhitened), prewhitened.shape)
        # print(aligned, prewhitened); exit()

        # XXX
        # print('read', image)
        # print('save to', os.path.expanduser(image) + '.new.jpg')
        # misc.imsave(os.path.expanduser(image) + '.aligned.jpg',aligned)
        # misc.imsave(os.path.expanduser(image) + '.prewhiten.jpg',prewhitened)
    images = np.stack(img_list)
    return images

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('image_files', type=str, nargs='+', help='Images to compare')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))


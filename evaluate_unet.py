"""Evaluation script for the DeepLab-ResNet network on the validation subset
   of PASCAL VOC dataset.
"""
from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time
import collections
import glob
import math

import tensorflow as tf
import numpy as np
import scipy as scp

from unet import unet 


# IMG_RGB_MEAN = np.array([104.42819731, 111.76448516, 125.9762497], dtype=np.float32)

DATA_DIRECTORY = '/Users/i859032/images/Red_Bull/512X512/data/val'
IGNORE_LABEL = 255
NUM_CLASSES = 2
IMG_SIZE = 512
# NUM_STEPS = 570# Number of images in the validation set.
RESTORE_FROM = '/Users/i859032/Desktop/jupyter/Image_Segmentation/pix2pix_AtoB/redbull/20170610'
Examples = collections.namedtuple("Examples", "inputs, targets, count")


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    # parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
    #                     help="Number of images in the validation set.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--is-training", action="store_false",
                        help="Whether to updates the running means and variances during the training.")    
    return parser.parse_args()


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def load_examples(args, Mean_Subtract=False, RGB_to_BGR=False, Normal=True):
    if args.data_dir is None or not os.path.exists(args.data_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(args.data_dir, "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(args.data_dir, "*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=args.is_training)
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])
        width = tf.shape(raw_input)[1] # [height, width, channels]
        print("width:", width)
        if Mean_Subtract:
            img = tf.cast(raw_input[:,:width//2,:], tf.float32) - IMG_RGB_MEAN
        else:
            if Normal:
                img = preprocess(tf.image.convert_image_dtype(raw_input[:,:width//2,:], dtype=tf.float32))
            else:
                img = tf.cast(raw_input[:,:width//2,:], tf.float32)

        if RGB_to_BGR: 
            img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
            a_images = tf.concat(axis=2, values=[img_b, img_g, img_r])
        else:
            a_images = img

        b_images = raw_input[:,width//2:,:]
        inputs, targets = [a_images, b_images]

    input_images = inputs
    targets1 = tf.squeeze(tf.to_int32(targets[:,:,2:]), squeeze_dims=[2])
    targets_one_hot = tf.one_hot(targets1, depth=args.num_classes, axis=-1)
    # targets2 = transform(targets_one_hot)
    target_images = tf.argmax(targets_one_hot, axis=-1)

    input_images.set_shape([IMG_SIZE, IMG_SIZE, 3])
    target_images.set_shape([IMG_SIZE, IMG_SIZE])

    # print(input_images.get_shape())
    # print(target_images.get_shape())


    # paths_batch, inputs_batch, targets_batch = tf.train.batch(
    #     [paths, input_images, target_images], 
    #     batch_size=1)

    return Examples(
        inputs=input_images,
        targets=target_images,
        count=len(input_paths),
    )


def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def main():
    
    

    """Create the model and start the evaluation process."""
    args = get_arguments()
    
    # Load reader.
    with tf.name_scope("create_inputs"):
        examples = load_examples(args)
        print("examples count = %d" % examples.count)

    image_batch, label_batch = tf.expand_dims(examples.inputs, dim=0), tf.expand_dims(examples.targets, dim=0)
    # net = DeepLabResNetModel({'data': image_batch}, is_training=False, num_classes=args.num_classes)

    print("image_batch:", image_batch.shape)
    print("label_batch:", label_batch.shape)
    with tf.variable_scope("generator") as scope:
        raw_output = unet(image_batch, args.num_classes, dropout_rate=0)

    # Which variables to load.
    # restore_var = tf.global_variables()
    
    # Predictions.
    raw_output = tf.argmax(raw_output, dimension=3)
    pred = tf.expand_dims(raw_output, dim=3) # Create 4-d tensor.
    
    # mIoU
    pred = tf.reshape(pred, [-1,])
    gt = tf.reshape(label_batch, [-1,])
    # weights = tf.cast(tf.less_equal(gt, args.num_classes - 1), tf.int32) # Ignoring all labels greater than or equal to n_classes.
    # mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=args.num_classes, weights=weights)
    mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=args.num_classes)


    loader = tf.train.Saver()

    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    with sess.as_default():
    
        sess.run(init)
        sess.run(tf.local_variables_initializer())
    
        # Load weights.
        checkpoint_file = tf.train.latest_checkpoint(args.restore_from)        
        loader.restore(sess, checkpoint_file)
        log_to = args.restore_from + '/log'
        if not os.path.exists(log_to):
            os.makedirs(log_to)

        
        print("Restored model parameters from {}".format(checkpoint_file))

        # Create queue coordinator.
        coord = tf.train.Coordinator()

        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        
        # Iterate over training steps.
        start_time = time.time()
        mIoU_list = []
        prev_mIoU = 0
        for step in range(examples.count):
#            output, _ , _ = sess.run([raw_output, pred, update_op])
            _ , _ = sess.run([pred, update_op])
            mIoU_value = sess.run(mIoU)
            cur_value = (step+1)*mIoU_value-step*prev_mIoU
            print(cur_value)
            mIoU_list.append(cur_value)
            prev_mIoU = mIoU_value
            if step % 100 == 0:
                print('step {:d}'.format(step))

#            scp.misc.imsave(log_to+'/eval_output_'+str(step)+'.png', output[0])

        duration = time.time() - start_time
        print('{:.3f} Time Consumed for {} pictures'.format(duration, examples.count))
        print('Mean IoU: {:.5f}'.format(sess.run(mIoU)))
        print('Mean IoU2: {:.5f}'.format(sum(mIoU_list)/len(mIoU_list)))
        coord.request_stop()
        coord.join(threads)
    
if __name__ == '__main__':
    main()

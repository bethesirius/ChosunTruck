#!/usr/bin/env python
import json
import cv2
import tensorflow.contrib.slim as slim
import datetime
import random
import time
import string
import argparse
import os
import threading
from scipy import misc
import tensorflow as tf
import numpy as np
try:
    from tensorflow.models.rnn import rnn_cell
except ImportError:
    rnn_cell = tf.nn.rnn_cell
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

random.seed(0)
np.random.seed(0)

from utils import train_utils, googlenet_load

@ops.RegisterGradient("Hungarian")
def _hungarian_grad(op, *args):
    return map(array_ops.zeros_like, op.inputs)

def build_lstm_inner(H, lstm_input):
    '''
    build lstm decoder
    '''
    lstm_cell = rnn_cell.BasicLSTMCell(H['lstm_size'], forget_bias=0.0, state_is_tuple=False)
    if H['num_lstm_layers'] > 1:
        lstm = rnn_cell.MultiRNNCell([lstm_cell] * H['num_lstm_layers'], state_is_tuple=False)
    else:
        lstm = lstm_cell

    batch_size = H['batch_size'] * H['grid_height'] * H['grid_width']
    state = tf.zeros([batch_size, lstm.state_size])

    outputs = []
    with tf.variable_scope('RNN', initializer=tf.random_uniform_initializer(-0.1, 0.1)):
        for time_step in range(H['rnn_len']):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            output, state = lstm(lstm_input, state)
            outputs.append(output)
    return outputs

def build_overfeat_inner(H, lstm_input):
    '''
    build simple overfeat decoder
    '''
    if H['rnn_len'] > 1:
        raise ValueError('rnn_len > 1 only supported with use_lstm == True')
    outputs = []
    initializer = tf.random_uniform_initializer(-0.1, 0.1)
    with tf.variable_scope('Overfeat', initializer=initializer):
        w = tf.get_variable('ip', shape=[H['later_feat_channels'], H['lstm_size']])
        outputs.append(tf.matmul(lstm_input, w))
    return outputs

def deconv(x, output_shape, channels):
    k_h = 2
    k_w = 2
    w = tf.get_variable('w_deconv', initializer=tf.random_normal_initializer(stddev=0.01),
                        shape=[k_h, k_w, channels[1], channels[0]])
    y = tf.nn.conv2d_transpose(x, w, output_shape, strides=[1, k_h, k_w, 1], padding='VALID')
    return y

def rezoom(H, pred_boxes, early_feat, early_feat_channels, w_offsets, h_offsets):
    '''
    Rezoom into a feature map at multiple interpolation points in a grid. 

    If the predicted object center is at X, len(w_offsets) == 3, and len(h_offsets) == 5,
    the rezoom grid will look as follows:

    [o o o]
    [o o o]
    [o X o]
    [o o o]
    [o o o]

    Where each letter indexes into the feature map with bilinear interpolation
    '''


    grid_size = H['grid_width'] * H['grid_height']
    outer_size = grid_size * H['batch_size']
    indices = []
    for w_offset in w_offsets:
        for h_offset in h_offsets:
            indices.append(train_utils.bilinear_select(H,
                                                       pred_boxes,
                                                       early_feat,
                                                       early_feat_channels,
                                                       w_offset, h_offset))

    interp_indices = tf.concat(0, indices)
    rezoom_features = train_utils.interp(early_feat,
                                         interp_indices,
                                         early_feat_channels)
    rezoom_features_r = tf.reshape(rezoom_features,
                                   [len(w_offsets) * len(h_offsets),
                                    outer_size,
                                    H['rnn_len'],
                                    early_feat_channels])
    rezoom_features_t = tf.transpose(rezoom_features_r, [1, 2, 0, 3])
    return tf.reshape(rezoom_features_t,
                      [outer_size,
                       H['rnn_len'],
                       len(w_offsets) * len(h_offsets) * early_feat_channels])

def build_forward(H, x, phase, reuse):
    '''
    Construct the forward model
    '''

    grid_size = H['grid_width'] * H['grid_height']
    outer_size = grid_size * H['batch_size']
    input_mean = 117.
    x -= input_mean
    cnn, early_feat = googlenet_load.model(x, H, reuse)
    early_feat_channels = H['early_feat_channels']
    early_feat = early_feat[:, :, :, :early_feat_channels]
    
    if H['deconv']:
        size = 3
        stride = 2
        pool_size = 5

        with tf.variable_scope("deconv", reuse=reuse):
            w = tf.get_variable('conv_pool_w', shape=[size, size, H['later_feat_channels'], H['later_feat_channels']],
                                initializer=tf.random_normal_initializer(stddev=0.01))
            cnn_s = tf.nn.conv2d(cnn, w, strides=[1, stride, stride, 1], padding='SAME')
            cnn_s_pool = tf.nn.avg_pool(cnn_s[:, :, :, :256], ksize=[1, pool_size, pool_size, 1],
                                        strides=[1, 1, 1, 1], padding='SAME')

            cnn_s_with_pool = tf.concat(3, [cnn_s_pool, cnn_s[:, :, :, 256:]])
            cnn_deconv = deconv(cnn_s_with_pool, output_shape=[H['batch_size'], H['grid_height'], H['grid_width'], 256], channels=[H['later_feat_channels'], 256])
            cnn = tf.concat(3, (cnn_deconv, cnn[:, :, :, 256:]))

    elif H['avg_pool_size'] > 1:
        pool_size = H['avg_pool_size']
        cnn1 = cnn[:, :, :, :700]
        cnn2 = cnn[:, :, :, 700:]
        cnn2 = tf.nn.avg_pool(cnn2, ksize=[1, pool_size, pool_size, 1],
                              strides=[1, 1, 1, 1], padding='SAME')
        cnn = tf.concat(3, [cnn1, cnn2])

    cnn = tf.reshape(cnn,
                     [H['batch_size'] * H['grid_width'] * H['grid_height'], H['later_feat_channels']])
    initializer = tf.random_uniform_initializer(-0.1, 0.1)
    with tf.variable_scope('decoder', reuse=reuse, initializer=initializer):
        scale_down = 0.01
        lstm_input = tf.reshape(cnn * scale_down, (H['batch_size'] * grid_size, H['later_feat_channels']))
        if H['use_lstm']:
            lstm_outputs = build_lstm_inner(H, lstm_input)
        else:
            lstm_outputs = build_overfeat_inner(H, lstm_input)

        pred_boxes = []
        pred_logits = []
        for k in range(H['rnn_len']):
            output = lstm_outputs[k]
            if phase == 'train':
                output = tf.nn.dropout(output, 0.5)
            box_weights = tf.get_variable('box_ip%d' % k,
                                          shape=(H['lstm_size'], 4))
            conf_weights = tf.get_variable('conf_ip%d' % k,
                                           shape=(H['lstm_size'], H['num_classes']))

            pred_boxes_step = tf.reshape(tf.matmul(output, box_weights) * 50,
                                         [outer_size, 1, 4])

            pred_boxes.append(pred_boxes_step)
            pred_logits.append(tf.reshape(tf.matmul(output, conf_weights),
                                         [outer_size, 1, H['num_classes']]))
 
        pred_boxes = tf.concat(1, pred_boxes)
        pred_logits = tf.concat(1, pred_logits)
        pred_logits_squash = tf.reshape(pred_logits,
                                        [outer_size * H['rnn_len'], H['num_classes']])
        pred_confidences_squash = tf.nn.softmax(pred_logits_squash)
        pred_confidences = tf.reshape(pred_confidences_squash,
                                      [outer_size, H['rnn_len'], H['num_classes']])

        if H['use_rezoom']:
            pred_confs_deltas = []
            pred_boxes_deltas = []
            w_offsets = H['rezoom_w_coords']
            h_offsets = H['rezoom_h_coords']
            num_offsets = len(w_offsets) * len(h_offsets)
            rezoom_features = rezoom(H, pred_boxes, early_feat, early_feat_channels, w_offsets, h_offsets)
            if phase == 'train':
                rezoom_features = tf.nn.dropout(rezoom_features, 0.5)
            for k in range(H['rnn_len']):
                delta_features = tf.concat(1, [lstm_outputs[k], rezoom_features[:, k, :] / 1000.])
                dim = 128
                delta_weights1 = tf.get_variable(
                                    'delta_ip1%d' % k,
                                    shape=[H['lstm_size'] + early_feat_channels * num_offsets, dim])
                # TODO: add dropout here ?
                ip1 = tf.nn.relu(tf.matmul(delta_features, delta_weights1))
                if phase == 'train':
                    ip1 = tf.nn.dropout(ip1, 0.5)
                delta_confs_weights = tf.get_variable(
                                    'delta_ip2%d' % k,
                                    shape=[dim, H['num_classes']])
                if H['reregress']:
                    delta_boxes_weights = tf.get_variable(
                                        'delta_ip_boxes%d' % k,
                                        shape=[dim, 4])
                    pred_boxes_deltas.append(tf.reshape(tf.matmul(ip1, delta_boxes_weights) * 5,
                                                        [outer_size, 1, 4]))
                scale = H.get('rezoom_conf_scale', 50) 
                pred_confs_deltas.append(tf.reshape(tf.matmul(ip1, delta_confs_weights) * scale,
                                                    [outer_size, 1, H['num_classes']]))
            pred_confs_deltas = tf.concat(1, pred_confs_deltas)
            if H['reregress']:
                pred_boxes_deltas = tf.concat(1, pred_boxes_deltas)
            return pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas

    return pred_boxes, pred_logits, pred_confidences

def build_forward_backward(H, x, phase, boxes, flags):
    '''
    Call build_forward() and then setup the loss functions
    '''

    grid_size = H['grid_width'] * H['grid_height']
    outer_size = grid_size * H['batch_size']
    reuse = {'train': None, 'test': True}[phase]
    if H['use_rezoom']:
        (pred_boxes, pred_logits,
         pred_confidences, pred_confs_deltas, pred_boxes_deltas) = build_forward(H, x, phase, reuse)
    else:
        pred_boxes, pred_logits, pred_confidences = build_forward(H, x, phase, reuse)
    with tf.variable_scope('decoder', reuse={'train': None, 'test': True}[phase]):
        outer_boxes = tf.reshape(boxes, [outer_size, H['rnn_len'], 4])
        outer_flags = tf.cast(tf.reshape(flags, [outer_size, H['rnn_len']]), 'int32')
        if H['use_lstm']:
            hungarian_module = tf.load_op_library('utils/hungarian/hungarian.so')
            assignments, classes, perm_truth, pred_mask = (
                hungarian_module.hungarian(pred_boxes, outer_boxes, outer_flags, H['solver']['hungarian_iou']))
        else:
            classes = tf.reshape(flags, (outer_size, 1))
            perm_truth = tf.reshape(outer_boxes, (outer_size, 1, 4))
            pred_mask = tf.reshape(tf.cast(tf.greater(classes, 0), 'float32'), (outer_size, 1, 1))
        true_classes = tf.reshape(tf.cast(tf.greater(classes, 0), 'int64'),
                                  [outer_size * H['rnn_len']])
        pred_logit_r = tf.reshape(pred_logits,
                                  [outer_size * H['rnn_len'], H['num_classes']])
        confidences_loss = (tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(pred_logit_r, true_classes))
            ) / outer_size * H['solver']['head_weights'][0]
        residual = tf.reshape(perm_truth - pred_boxes * pred_mask,
                              [outer_size, H['rnn_len'], 4])
        boxes_loss = tf.reduce_sum(tf.abs(residual)) / outer_size * H['solver']['head_weights'][1]
        if H['use_rezoom']:
            if H['rezoom_change_loss'] == 'center':
                error = (perm_truth[:, :, 0:2] - pred_boxes[:, :, 0:2]) / tf.maximum(perm_truth[:, :, 2:4], 1.)
                square_error = tf.reduce_sum(tf.square(error), 2)
                inside = tf.reshape(tf.to_int64(tf.logical_and(tf.less(square_error, 0.2**2), tf.greater(classes, 0))), [-1])
            elif H['rezoom_change_loss'] == 'iou':
                iou = train_utils.iou(train_utils.to_x1y1x2y2(tf.reshape(pred_boxes, [-1, 4])),
                                      train_utils.to_x1y1x2y2(tf.reshape(perm_truth, [-1, 4])))
                inside = tf.reshape(tf.to_int64(tf.greater(iou, 0.5)), [-1])
            else:
                assert H['rezoom_change_loss'] == False
                inside = tf.reshape(tf.to_int64((tf.greater(classes, 0))), [-1])
            new_confs = tf.reshape(pred_confs_deltas, [outer_size * H['rnn_len'], H['num_classes']])
            delta_confs_loss = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(new_confs, inside)) / outer_size * H['solver']['head_weights'][0] * 0.1

            pred_logits_squash = tf.reshape(new_confs,
                                            [outer_size * H['rnn_len'], H['num_classes']])
            pred_confidences_squash = tf.nn.softmax(pred_logits_squash)
            pred_confidences = tf.reshape(pred_confidences_squash,
                                      [outer_size, H['rnn_len'], H['num_classes']])
            loss = confidences_loss + boxes_loss + delta_confs_loss
            if H['reregress']:
                delta_residual = tf.reshape(perm_truth - (pred_boxes + pred_boxes_deltas) * pred_mask,
                                            [outer_size, H['rnn_len'], 4])
                delta_boxes_loss = (tf.reduce_sum(tf.minimum(tf.square(delta_residual), 10. ** 2)) / 
                               outer_size * H['solver']['head_weights'][1] * 0.03)
                boxes_loss = delta_boxes_loss

                tf.summary.histogram(phase + '/delta_hist0_x', pred_boxes_deltas[:, 0, 0])
                tf.summary.histogram(phase + '/delta_hist0_y', pred_boxes_deltas[:, 0, 1])
                tf.summary.histogram(phase + '/delta_hist0_w', pred_boxes_deltas[:, 0, 2])
                tf.summary.histogram(phase + '/delta_hist0_h', pred_boxes_deltas[:, 0, 3])
                loss += delta_boxes_loss
        else:
            loss = confidences_loss + boxes_loss

    return pred_boxes, pred_confidences, loss, confidences_loss, boxes_loss

def build(H, q):
    '''
    Build full model for training, including forward / backward passes,
    optimizers, and summary statistics.
    '''
    arch = H
    solver = H["solver"]

    os.environ['CUDA_VISIBLE_DEVICES'] = str(solver.get('gpu', ''))

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    gpu_options = tf.GPUOptions()
    config = tf.ConfigProto(gpu_options=gpu_options)

    learning_rate = tf.placeholder(tf.float32)
    if solver['opt'] == 'RMS':
        opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                        decay=0.9, epsilon=solver['epsilon'])
    elif solver['opt'] == 'Adam':
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                        epsilon=solver['epsilon'])
    elif solver['opt'] == 'SGD':
        opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    else:
        raise ValueError('Unrecognized opt type')
    loss, accuracy, confidences_loss, boxes_loss = {}, {}, {}, {}
    for phase in ['train', 'test']:
        # generate predictions and losses from forward pass
        x, confidences, boxes = q[phase].dequeue_many(arch['batch_size'])
        flags = tf.argmax(confidences, 3)


        grid_size = H['grid_width'] * H['grid_height']

        (pred_boxes, pred_confidences,
         loss[phase], confidences_loss[phase],
         boxes_loss[phase]) = build_forward_backward(H, x, phase, boxes, flags)
        pred_confidences_r = tf.reshape(pred_confidences, [H['batch_size'], grid_size, H['rnn_len'], arch['num_classes']])
        pred_boxes_r = tf.reshape(pred_boxes, [H['batch_size'], grid_size, H['rnn_len'], 4])


        # Set up summary operations for tensorboard
        a = tf.equal(tf.argmax(confidences[:, :, 0, :], 2), tf.argmax(pred_confidences_r[:, :, 0, :], 2))
        accuracy[phase] = tf.reduce_mean(tf.cast(a, 'float32'), name=phase+'/accuracy')

        if phase == 'train':
            global_step = tf.Variable(0, trainable=False)

            tvars = tf.trainable_variables()
            if H['clip_norm'] <= 0:
                grads = tf.gradients(loss['train'], tvars)
            else:
                grads, norm = tf.clip_by_global_norm(tf.gradients(loss['train'], tvars), H['clip_norm'])
            train_op = opt.apply_gradients(zip(grads, tvars), global_step=global_step)
        elif phase == 'test':
            moving_avg = tf.train.ExponentialMovingAverage(0.95)
            smooth_op = moving_avg.apply([accuracy['train'], accuracy['test'],
                                          confidences_loss['train'], boxes_loss['train'],
                                          confidences_loss['test'], boxes_loss['test'],
                                          ])

            for p in ['train', 'test']:
                tf.summary.scalar('%s/accuracy' % p, accuracy[p])
                tf.summary.scalar('%s/accuracy/smooth' % p, moving_avg.average(accuracy[p]))
                tf.summary.scalar("%s/confidences_loss" % p, confidences_loss[p])
                tf.summary.scalar("%s/confidences_loss/smooth" % p,
                    moving_avg.average(confidences_loss[p]))
                tf.summary.scalar("%s/regression_loss" % p, boxes_loss[p])
                tf.summary.scalar("%s/regression_loss/smooth" % p,
                    moving_avg.average(boxes_loss[p]))

        if phase == 'test':
            test_image = x
            # show ground truth to verify labels are correct
            test_true_confidences = confidences[0, :, :, :]
            test_true_boxes = boxes[0, :, :, :]

            # show predictions to visualize training progress
            test_pred_confidences = pred_confidences_r[0, :, :, :]
            test_pred_boxes = pred_boxes_r[0, :, :, :]

            def log_image(np_img, np_confidences, np_boxes, np_global_step, pred_or_true):
                
                merged = train_utils.add_rectangles(H, np_img, np_confidences, np_boxes,
                                                    use_stitching=True,
                                                    rnn_len=H['rnn_len'])[0]
                
                num_images = 10
                img_path = os.path.join(H['save_dir'], '%s_%s.jpg' % ((np_global_step / H['logging']['display_iter']) % num_images, pred_or_true))
                misc.imsave(img_path, merged)
                return merged

            pred_log_img = tf.py_func(log_image,
                                      [test_image, test_pred_confidences, test_pred_boxes, global_step, 'pred'],
                                      [tf.float32])
            true_log_img = tf.py_func(log_image,
                                      [test_image, test_true_confidences, test_true_boxes, global_step, 'true'],
                                      [tf.float32])
            tf.image_summary(phase + '/pred_boxes', tf.pack(pred_log_img),max_images=10)
            tf.image_summary(phase + '/true_boxes', tf.pack(true_log_img),max_images=10)

    summary_op = tf.summary.merge_all()

    return (config, loss, accuracy, summary_op, train_op,
            smooth_op, global_step, learning_rate)


def train(H, test_images):
    '''
    Setup computation graph, run 2 prefetch data threads, and then run the main loop
    '''

    if not os.path.exists(H['save_dir']): os.makedirs(H['save_dir'])

    ckpt_file = H['save_dir'] + '/save.ckpt'
    with open(H['save_dir'] + '/hypes.json', 'w') as f:
        json.dump(H, f, indent=4)

    x_in = tf.placeholder(tf.float32)
    confs_in = tf.placeholder(tf.float32)
    boxes_in = tf.placeholder(tf.float32)
    q = {}
    enqueue_op = {}
    for phase in ['train', 'test']:
        dtypes = [tf.float32, tf.float32, tf.float32]
        grid_size = H['grid_width'] * H['grid_height']
        shapes = (
            [H['image_height'], H['image_width'], 3],
            [grid_size, H['rnn_len'], H['num_classes']],
            [grid_size, H['rnn_len'], 4],
            )
        q[phase] = tf.FIFOQueue(capacity=30, dtypes=dtypes, shapes=shapes)
        enqueue_op[phase] = q[phase].enqueue((x_in, confs_in, boxes_in))

    def make_feed(d):
        return {x_in: d['image'], confs_in: d['confs'], boxes_in: d['boxes'],
                learning_rate: H['solver']['learning_rate']}

    def thread_loop(sess, enqueue_op, phase, gen):
        for d in gen:
            sess.run(enqueue_op[phase], feed_dict=make_feed(d))

    (config, loss, accuracy, summary_op, train_op,
     smooth_op, global_step, learning_rate) = build(H, q)

    saver = tf.train.Saver(max_to_keep=None)
    writer = tf.train.SummaryWriter(
        logdir=H['save_dir'],
        flush_secs=10
    )

    with tf.Session(config=config) as sess:
        tf.train.start_queue_runners(sess=sess)
        for phase in ['train', 'test']:
            # enqueue once manually to avoid thread start delay
            gen = train_utils.load_data_gen(H, phase, jitter=H['solver']['use_jitter'])
            d = gen.next()
            sess.run(enqueue_op[phase], feed_dict=make_feed(d))
            t = threading.Thread(target=thread_loop,
                                 args=(sess, enqueue_op, phase, gen))
            t.daemon = True
            t.start()

        tf.set_random_seed(H['solver']['rnd_seed'])
        sess.run(tf.initialize_all_variables())
        writer.add_graph(sess.graph)
        weights_str = H['solver']['weights']
        if len(weights_str) > 0:
            print('Restoring from: %s' % weights_str)
            saver.restore(sess, weights_str)
        else:
            init_fn = slim.assign_from_checkpoint_fn(
                  '%s/data/%s' % (os.path.dirname(os.path.realpath(__file__)), H['slim_ckpt']),
                  [x for x in tf.all_variables() if x.name.startswith(H['slim_basename']) and H['solver']['opt'] not in x.name])
            #init_fn = slim.assign_from_checkpoint_fn(
                  #'%s/data/inception_v1.ckpt' % os.path.dirname(os.path.realpath(__file__)),
                  #[x for x in tf.all_variables() if x.name.startswith('InceptionV1') and not H['solver']['opt'] in x.name])
            init_fn(sess)

        # train model for N iterations
        start = time.time()
        max_iter = H['solver'].get('max_iter', 100000)
        for i in xrange(max_iter):
            display_iter = H['logging']['display_iter']
            adjusted_lr = (H['solver']['learning_rate'] *
                           0.5 ** max(0, (i / H['solver']['learning_rate_step']) - 2))
            lr_feed = {learning_rate: adjusted_lr}

            if i % display_iter != 0:
                # train network
                batch_loss_train, _ = sess.run([loss['train'], train_op], feed_dict=lr_feed)
            else:
                # test network every N iterations; log additional info
                if i > 0:
                    dt = (time.time() - start) / (H['batch_size'] * display_iter)
                start = time.time()
                (train_loss, test_accuracy, summary_str,
                    _, _) = sess.run([loss['train'], accuracy['test'],
                                      summary_op, train_op, smooth_op,
                                     ], feed_dict=lr_feed)
                writer.add_summary(summary_str, global_step=global_step.eval())
                print_str = string.join([
                    'Step: %d',
                    'lr: %f',
                    'Train Loss: %.2f',
                    'Softmax Test Accuracy: %.1f%%',
                    'Time/image (ms): %.1f'
                ], ', ')
                print(print_str %
                      (i, adjusted_lr, train_loss,
                       test_accuracy * 100, dt * 1000 if i > 0 else 0))

            if global_step.eval() % H['logging']['save_iter'] == 0 or global_step.eval() == max_iter - 1:
                saver.save(sess, ckpt_file, global_step=global_step)


def main():
    '''
    Parse command line arguments and return the hyperparameter dictionary H.
    H first loads the --hypes hypes.json file and is further updated with
    additional arguments as needed.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=None, type=str)
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--hypes', required=True, type=str)
    parser.add_argument('--logdir', default='output', type=str)
    args = parser.parse_args()
    with open(args.hypes, 'r') as f:
        H = json.load(f)
    if args.gpu is not None:
        H['solver']['gpu'] = args.gpu
    if len(H.get('exp_name', '')) == 0:
        H['exp_name'] = args.hypes.split('/')[-1].replace('.json', '')
    H['save_dir'] = args.logdir + '/%s_%s' % (H['exp_name'],
        datetime.datetime.now().strftime('%Y_%m_%d_%H.%M'))
    if args.weights is not None:
        H['solver']['weights'] = args.weights
    train(H, test_images=[])

if __name__ == '__main__':
    main()

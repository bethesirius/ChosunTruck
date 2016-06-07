#!/usr/bin/env python
# 0.1 initialization on all LSTM
# average pooling on Z
# decrease regression multiplier
# 2 layer lstm
# TODO:
# reregression
# dropout on reclassification
# 3x3 or 2x3 reclassification
import json
import datetime
import random
import time
import string
import argparse
import os
import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn_cell
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

random.seed(0)
np.random.seed(0)

from utils import train_utils
from utils import googlenet_load

def build_lstm_inner(lstm_input, H):
    '''
    build lstm decoder
    '''
    lstm_cell = rnn_cell.BasicLSTMCell(H['arch']['lstm_size'], forget_bias=0.0)
    if H['arch']['num_lstm_layers'] > 1:
        lstm = rnn_cell.MultiRNNCell([lstm_cell] * H['arch']['num_lstm_layers'])
    else:
        lstm = lstm_cell

    batch_size = H['arch']['batch_size'] * H['arch']['grid_height'] * H['arch']['grid_width']
    state = tf.zeros([batch_size, lstm.state_size])

    outputs = []
    with tf.variable_scope('RNN', initializer=tf.random_uniform_initializer(-0.1, 0.1)):
        for time_step in range(H['arch']['rnn_len']):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            output, state = lstm(lstm_input, state)
            outputs.append(output)
    return outputs

def build_overfeat_inner(lstm_input, H):
    '''
    build simple overfeat decoder
    '''
    if H['arch']['rnn_len'] > 1:
        raise ValueError('rnn_len > 1 only supported with use_lstm == True')
    outputs = []
    with tf.variable_scope('Overfeat', initializer=tf.random_uniform_initializer(-0.1, 0.1)):
        w = tf.get_variable('ip', shape=[1024, H['arch']['lstm_size']])
        outputs.append(tf.matmul(lstm_input, w))
    return outputs

def to_idx(vec, w_shape):
    '''
    vec = (idn, idh, idw)
    w_shape = [n, h, w, c]
    '''
    return vec[:, 2] + w_shape[2] * (vec[:, 1] + w_shape[1] * vec[:, 0])

def interp(w, i, channel_dim):
    '''
    Input:
        w: A 4D block tensor of shape (n, h, w, c)
        i: A list of 3-tuples [(x_1, y_1, z_1), (x_2, y_2, z_2), ...],
            each having type (int, float, float)
 
        The 4D block represents a batch of 3D image feature volumes with c channels.
        The input i is a list of points  to index into w via interpolation. Direct
        indexing is not possible due to y_1 and z_1 being float values.
    Output:
        A list of the values: [
            w[x_1, y_1, z_1, :]
            w[x_2, y_2, z_2, :]
            ...
            w[x_k, y_k, z_k, :]
        ]
        of the same length == len(i)
    '''
    w_as_vector = tf.reshape(w, [-1, channel_dim]) # gather expects w to be 1-d
    upper_l = tf.to_int32(tf.concat(1, [i[:, 0:1], tf.floor(i[:, 1:2]), tf.floor(i[:, 2:3])]))
    upper_r = tf.to_int32(tf.concat(1, [i[:, 0:1], tf.floor(i[:, 1:2]), tf.ceil(i[:, 2:3])]))
    lower_l = tf.to_int32(tf.concat(1, [i[:, 0:1], tf.ceil(i[:, 1:2]), tf.floor(i[:, 2:3])]))
    lower_r = tf.to_int32(tf.concat(1, [i[:, 0:1], tf.ceil(i[:, 1:2]), tf.ceil(i[:, 2:3])]))

    upper_l_idx = to_idx(upper_l, tf.shape(w))
    upper_r_idx = to_idx(upper_r, tf.shape(w))
    lower_l_idx = to_idx(lower_l, tf.shape(w))
    lower_r_idx = to_idx(lower_r, tf.shape(w))
 
    upper_l_value = tf.gather(w_as_vector, upper_l_idx)
    upper_r_value = tf.gather(w_as_vector, upper_r_idx)
    lower_l_value = tf.gather(w_as_vector, lower_l_idx)
    lower_r_value = tf.gather(w_as_vector, lower_r_idx)
 
    alpha_lr = tf.expand_dims(i[:, 2] - tf.floor(i[:, 2]), 1)
    alpha_ud = tf.expand_dims(i[:, 1] - tf.floor(i[:, 1]), 1)
 
    upper_value = (1 - alpha_lr) * upper_l_value + (alpha_lr) * upper_r_value
    lower_value = (1 - alpha_lr) * lower_l_value + (alpha_lr) * lower_r_value
    value = (1 - alpha_ud) * upper_value + (alpha_ud) * lower_value
    return value

def bilinear_select(H, pred_boxes, early_feat, early_feat_channels, w_offset, h_offset):
    '''
    Function used for rezooming high level feature maps. Uses bilinear interpolation
    to select all channels at index (x, y) for a high level feature map, where x and y are floats.
    '''
    grid_size = H['arch']['grid_width'] * H['arch']['grid_height']
    outer_size = grid_size * H['arch']['batch_size']

    fine_stride = 8. # pixels per 60x80 grid cell in 480x640 image
    coarse_stride = H['arch']['region_size'] # pixels per 15x20 grid cell in 480x640 image
    batch_ids = []
    x_offsets = []
    y_offsets = []
    for n in range(H['arch']['batch_size']):
        for i in range(H['arch']['grid_height']):
            for j in range(H['arch']['grid_width']):
                for k in range(H['arch']['rnn_len']):
                    batch_ids.append([n])
                    x_offsets.append([coarse_stride / 2. + coarse_stride * j])
                    y_offsets.append([coarse_stride / 2. + coarse_stride * i])

    batch_ids = tf.constant(batch_ids)
    x_offsets = tf.constant(x_offsets)
    y_offsets = tf.constant(y_offsets)

    pred_boxes_r = tf.reshape(pred_boxes, [outer_size * H['arch']['rnn_len'], 4])
    scale_factor = coarse_stride / fine_stride # scale difference between 15x20 and 60x80 features

    pred_x_center = (pred_boxes_r[:, 0:1] + w_offset * pred_boxes_r[:, 2:3] + x_offsets) / fine_stride
    pred_x_center_clip = tf.clip_by_value(pred_x_center,
                                     0,
                                     scale_factor * H['arch']['grid_width'] - 1)
    pred_y_center = (pred_boxes_r[:, 1:2] + h_offset * pred_boxes_r[:, 3:4] + y_offsets) / fine_stride
    pred_y_center_clip = tf.clip_by_value(pred_y_center,
                                          0,
                                          scale_factor * H['arch']['grid_height'] - 1)

    interp_indices = tf.concat(1, [tf.to_float(batch_ids), pred_y_center_clip, pred_x_center_clip])
    return interp_indices

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


    grid_size = H['arch']['grid_width'] * H['arch']['grid_height']
    outer_size = grid_size * H['arch']['batch_size']
    indices = []
    for w_offset in w_offsets:
        for h_offset in h_offsets:
            indices.append(bilinear_select(H, pred_boxes, early_feat, early_feat_channels, w_offset, h_offset))

    interp_indices = tf.concat(0, indices)
    rezoom_features = interp(early_feat, interp_indices, early_feat_channels)
    rezoom_features_r = tf.reshape(rezoom_features,
                                      [len(w_offsets) * len(h_offsets), outer_size, H['arch']['rnn_len'], early_feat_channels])
    rezoom_features_t = tf.transpose(rezoom_features_r, [1, 2, 0, 3])
    rezoom_features_t_r = tf.reshape(rezoom_features_t,
                                          [outer_size, H['arch']['rnn_len'], len(w_offsets) * len(h_offsets) * early_feat_channels])

    return rezoom_features_t_r

def build_forward(H, x, googlenet, phase, reuse):
    '''
    Construct the forward model
    '''

    grid_size = H['arch']['grid_width'] * H['arch']['grid_height']
    outer_size = grid_size * H['arch']['batch_size']
    input_mean = 117.
    x -= input_mean
    global early_feat
    Z, early_feat, _ = googlenet_load.model(x, googlenet, H)
    early_feat_channels = H['arch']['early_feat_channels']
    early_feat = early_feat[:, :, :, :early_feat_channels]
    
    if H['arch']['avg_pool_size'] > 1:
        pool_size = H['arch']['avg_pool_size']
        Z1 = Z[:, :, :, :700]
        Z2 = Z[:, :, :, 700:]
        Z2 = tf.nn.avg_pool(Z2, ksize=[1, pool_size, pool_size, 1], strides=[1, 1, 1, 1], padding='SAME')
        Z = tf.concat(3, [Z1, Z2])
    Z = tf.reshape(Z, [H['arch']['batch_size'] * H['arch']['grid_width'] * H['arch']['grid_height'], 1024])
    with tf.variable_scope('decoder', reuse=reuse):
        scale_down = 0.01
        lstm_input = tf.reshape(Z * scale_down, (H['arch']['batch_size'] * grid_size, 1024))
        if H['arch']['use_lstm']:
            lstm_outputs = build_lstm_inner(lstm_input, H)
        else:
            lstm_outputs = build_overfeat_inner(lstm_input, H)

        pred_boxes = []
        pred_logits = []
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        for k in range(H['arch']['rnn_len']):
            output = lstm_outputs[k]
            if phase == 'train':
                output = tf.nn.dropout(output, 0.5)
            box_weights = tf.get_variable('box_ip%d' % k,
                                          shape=(H['arch']['lstm_size'], 4),
                                          initializer=initializer)
            conf_weights = tf.get_variable('conf_ip%d' % k,
                                           shape=(H['arch']['lstm_size'], 2),
                                           initializer=initializer)

            pred_boxes_step = tf.reshape(tf.matmul(output, box_weights) * 50,
                                         [outer_size, 1, 4])

            pred_boxes.append(pred_boxes_step)
            pred_logits.append(tf.reshape(tf.matmul(output, conf_weights),
                                         [outer_size, 1, 2]))
 
        pred_boxes = tf.concat(1, pred_boxes)
        pred_logits = tf.concat(1, pred_logits)
        pred_logits_squash = tf.reshape(pred_logits,
                                        [outer_size * H['arch']['rnn_len'], 2])
        pred_confidences_squash = tf.nn.softmax(pred_logits_squash)
        pred_confidences = tf.reshape(pred_confidences_squash,
                                      [outer_size, H['arch']['rnn_len'], 2])

        if H['arch']['use_rezoom']:
            pred_confs_deltas = []
            pred_boxes_deltas = []
            w_offsets = H['arch']['rezoom_w_coords']
            h_offsets = H['arch']['rezoom_h_coords']
            num_offsets = len(w_offsets) * len(h_offsets)
            rezoom_features = rezoom(H, pred_boxes, early_feat, early_feat_channels, w_offsets, h_offsets)
            if phase == 'train':
                rezoom_features = tf.nn.dropout(rezoom_features, 0.5)
            for k in range(H['arch']['rnn_len']):
                delta_features = tf.concat(1, [lstm_outputs[k], rezoom_features[:, k, :] / 1000.])
                dim = 128
                delta_weights1 = tf.get_variable(
                                    'delta_ip1%d' % k,
                                    shape=[H['arch']['lstm_size'] + early_feat_channels * num_offsets, dim],
                                    initializer=initializer)
                # TODO: add dropout here ?
                ip1 = tf.nn.relu(tf.matmul(delta_features, delta_weights1))
                if phase == 'train':
                    ip1 = tf.nn.dropout(ip1, 0.5)
                delta_confs_weights = tf.get_variable(
                                    'delta_ip2%d' % k,
                                    shape=[dim, 2],
                                    initializer=initializer)
                if H['arch']['reregress']:
                    delta_boxes_weights = tf.get_variable(
                                        'delta_ip_boxes%d' % k,
                                        shape=[dim, 4],
                                        initializer=initializer)
                    pred_boxes_deltas.append(tf.reshape(tf.matmul(ip1, delta_boxes_weights) * 5,
                                                        [outer_size, 1, 4]))
                scale = H['arch'].get('rezoom_conf_scale', 50) 
                pred_confs_deltas.append(tf.reshape(tf.matmul(ip1, delta_confs_weights) * scale,
                                                    [outer_size, 1, 2]))
            pred_confs_deltas = tf.concat(1, pred_confs_deltas)
            if H['arch']['reregress']:
                pred_boxes_deltas = tf.concat(1, pred_boxes_deltas)
            return pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas

    return pred_boxes, pred_logits, pred_confidences

@ops.RegisterGradient("Hungarian")
def _hungarian_grad(op, *args):
    return map(array_ops.zeros_like, op.inputs)

def build_forward_backward(H, x, googlenet, phase, boxes, flags):
    '''
    Call build_forward() and then setup the loss functions
    '''

    grid_size = H['arch']['grid_width'] * H['arch']['grid_height']
    outer_size = grid_size * H['arch']['batch_size']
    reuse = {'train': None, 'test': True}[phase]
    if H['arch']['use_rezoom']:
        (pred_boxes, pred_logits,
         pred_confidences, pred_confs_deltas, pred_boxes_deltas) = build_forward(H, x, googlenet, phase, reuse)
    else:
        pred_boxes, pred_logits, pred_confidences = build_forward(H, x, googlenet, phase, reuse)
    with tf.variable_scope('decoder', reuse={'train': None, 'test': True}[phase]):
        outer_boxes = tf.reshape(boxes, [outer_size, H['arch']['rnn_len'], 4])
        outer_flags = tf.cast(tf.reshape(flags, [outer_size, H['arch']['rnn_len']]), 'int32')
        if H['arch']['use_lstm']:
            assignments, classes, perm_truth, pred_mask = (
                tf.user_ops.hungarian(pred_boxes, outer_boxes, outer_flags, H['solver']['hungarian_iou']))
        else:
            classes = tf.reshape(flags, (outer_size, 1))
            perm_truth = tf.reshape(outer_boxes, (outer_size, 1, 4))
            pred_mask = tf.reshape(tf.cast(tf.greater(classes, 0), 'float32'), (outer_size, 1, 1))
        true_classes = tf.reshape(tf.cast(tf.greater(classes, 0), 'int64'),
                                  [outer_size * H['arch']['rnn_len']])
        pred_logit_r = tf.reshape(pred_logits,
                                  [outer_size * H['arch']['rnn_len'], 2])
        confidences_loss = (tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(pred_logit_r, true_classes))
            ) / outer_size * H['solver']['head_weights'][0]
        residual = tf.reshape(perm_truth - pred_boxes * pred_mask,
                              [outer_size, H['arch']['rnn_len'], 4])
        boxes_loss = tf.reduce_sum(tf.abs(residual)) / outer_size * H['solver']['head_weights'][1]
        if H['arch']['use_rezoom']:
            if H['arch']['rezoom_change_loss'] == 'center':
                error = (perm_truth[:, :, 0:2] - pred_boxes[:, :, 0:2]) / tf.maximum(perm_truth[:, :, 2:4], 1.)
                square_error = tf.reduce_sum(tf.square(error), 2)
                inside = tf.reshape(tf.to_int64(tf.logical_and(tf.less(square_error, 0.2**2), tf.greater(classes, 0))), [-1])
            elif H['arch']['rezoom_change_loss'] == 'iou':
                iou = train_utils.iou(train_utils.to_x1y1x2y2(tf.reshape(pred_boxes, [-1, 4])),
                                      train_utils.to_x1y1x2y2(tf.reshape(perm_truth, [-1, 4])))
                inside = tf.reshape(tf.to_int64(tf.greater(iou, 0.5)), [-1])
            else:
                assert H['arch']['rezoom_change_loss'] == False
                inside = tf.reshape(tf.to_int64((tf.greater(classes, 0))), [-1])
            new_confs = tf.reshape(pred_confs_deltas, [outer_size * H['arch']['rnn_len'], 2])
            delta_confs_loss = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(new_confs, inside)) / outer_size * H['solver']['head_weights'][0] * 0.1

            use_orig_conf = H['solver'].get('use_orig_confs', False)
            if not use_orig_conf:
                confidences_loss = delta_confs_loss
            pred_logits_squash = tf.reshape(new_confs,
                                            [outer_size * H['arch']['rnn_len'], 2])
            pred_confidences_squash = tf.nn.softmax(pred_logits_squash)
            pred_confidences = tf.reshape(pred_confidences_squash,
                                      [outer_size, H['arch']['rnn_len'], 2])
            loss = confidences_loss + boxes_loss + delta_confs_loss
            confidences_loss = delta_confs_loss
            if H['arch']['reregress']:
                delta_residual = tf.reshape(perm_truth - (pred_boxes + pred_boxes_deltas) * pred_mask,
                                            [outer_size, H['arch']['rnn_len'], 4])
                delta_boxes_loss = (tf.reduce_sum(tf.minimum(tf.square(delta_residual), 10. ** 2)) / 
                               outer_size * H['solver']['head_weights'][1] * 0.03)
                boxes_loss = delta_boxes_loss

                tf.histogram_summary(phase + '/delta_hist0_x', pred_boxes_deltas[:, 0, 0])
                tf.histogram_summary(phase + '/delta_hist0_y', pred_boxes_deltas[:, 0, 1])
                tf.histogram_summary(phase + '/delta_hist0_w', pred_boxes_deltas[:, 0, 2])
                tf.histogram_summary(phase + '/delta_hist0_h', pred_boxes_deltas[:, 0, 3])
                loss += delta_boxes_loss
        else:
            loss = confidences_loss + boxes_loss

    return pred_boxes, pred_confidences, loss, confidences_loss, boxes_loss

def build(H, q):
    '''
    Build full model for training, including forward / backward passes,
    optimizers, and summary statistics.
    '''
    arch = H['arch']
    solver = H["solver"]

    os.environ['CUDA_VISIBLE_DEVICES'] = str(solver['gpu'])

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    gpu_options = tf.GPUOptions()
    config = tf.ConfigProto(gpu_options=gpu_options)

    encoder_net = googlenet_load.init(H, config)
    W_norm = encoder_net['W_norm']

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


        grid_size = H['arch']['grid_width'] * H['arch']['grid_height']

        (pred_boxes, pred_confidences,
         loss[phase], confidences_loss[phase],
         boxes_loss[phase]) = build_forward_backward(H, x, encoder_net, phase, boxes, flags)
        pred_confidences_r = tf.reshape(pred_confidences, [H['arch']['batch_size'], grid_size, H['arch']['rnn_len'], arch['num_classes']])
        pred_boxes_r = tf.reshape(pred_boxes, [H['arch']['batch_size'], grid_size, H['arch']['rnn_len'], 4])


        # Set up summary operations for tensorboard
        a = tf.equal(tf.argmax(confidences[:, :, 0, :], 2), tf.argmax(pred_confidences_r[:, :, 0, :], 2))
        accuracy[phase] = tf.reduce_mean(tf.cast(a, 'float32'), name=phase+'/accuracy')

        if phase == 'train':
            global_step = tf.Variable(0, trainable=False)

            tvars = tf.trainable_variables()
            if H['arch']['clip_norm'] <= 0:
                grads = tf.gradients(loss['train'], tvars)
            else:
                grads, norm = tf.clip_by_global_norm(tf.gradients(loss['train'], tvars), H['arch']['clip_norm'])
            train_op = opt.apply_gradients(zip(grads, tvars), global_step=global_step)
        elif phase == 'test':
            moving_avg = tf.train.ExponentialMovingAverage(0.95)
            smooth_op = moving_avg.apply([accuracy['train'], accuracy['test'],
                                          confidences_loss['train'], boxes_loss['train'],
                                          confidences_loss['test'], boxes_loss['test'],
                                          ])

            for p in ['train', 'test']:
                tf.scalar_summary('%s/accuracy' % p, accuracy[p])
                tf.scalar_summary('%s/accuracy/smooth' % p, moving_avg.average(accuracy[p]))
                tf.scalar_summary("%s/confidences_loss" % p, confidences_loss[p])
                tf.scalar_summary("%s/confidences_loss/smooth" % p,
                    moving_avg.average(confidences_loss[p]))
                tf.scalar_summary("%s/regression_loss" % p, boxes_loss[p])
                tf.scalar_summary("%s/regression_loss/smooth" % p,
                    moving_avg.average(boxes_loss[p]))

        if phase == 'test':
            test_image = x
            # show ground truth to verify labels are correct
            test_true_confidences = confidences[0, :, :, :]
            test_true_boxes = boxes[0, :, :, :]

            # show predictions to visualize training progress
            test_pred_confidences = pred_confidences_r[0, :, :, :]
            test_pred_boxes = pred_boxes_r[0, :, :, :]

    summary_op = tf.merge_all_summaries()

    return (config, loss, accuracy, summary_op, train_op, W_norm,
            test_image, test_pred_boxes, test_pred_confidences,
            test_true_boxes, test_true_confidences, smooth_op,
            global_step, learning_rate, encoder_net)


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
        grid_size = H['arch']['grid_width'] * H['arch']['grid_height']
        shapes = (
            [H['arch']['image_height'], H['arch']['image_width'], 3],
            [grid_size, H['arch']['rnn_len'], H['arch']['num_classes']],
            [grid_size, H['arch']['rnn_len'], 4],
            )
        q[phase] = tf.FIFOQueue(capacity=30, dtypes=dtypes, shapes=shapes)
        enqueue_op[phase] = q[phase].enqueue((x_in, confs_in, boxes_in))

    def make_feed(d):
        return {x_in: d['image'], confs_in: d['confs'], boxes_in: d['boxes'],
                learning_rate: H['solver']['learning_rate']}

    def my_loop(coord, sess, enqueue_op, phase, gen):
        for d in gen:
            try:
                sess.run(enqueue_op[phase], feed_dict=make_feed(d))
            except tf.errors.CancelledError:
                print('Cancelling data input queues\n')
                break

    (config, loss, accuracy, summary_op, train_op, W_norm,
     test_image, test_pred_boxes, test_pred_confidences,
     test_true_boxes, test_true_confidences,
     smooth_op, global_step, learning_rate, encoder_net) = build(H, q)

    saver = tf.train.Saver(max_to_keep=None)
    writer = tf.train.SummaryWriter(
        logdir=H['save_dir'],
        flush_secs=10
    )

    test_image_to_log = tf.placeholder(tf.uint8,
                                       [H['arch']['image_height'], H['arch']['image_width'], 3])
    log_image_name = tf.placeholder(tf.string)
    log_image = tf.image_summary(log_image_name, tf.expand_dims(test_image_to_log, 0))

    coord = tf.train.Coordinator()
    with tf.Session(config=config) as sess:
        threads = []
        for phase in ['train', 'test']:
            # enqueue once manually to avoid thread start delay
            gen = train_utils.load_data_gen(H, phase, jitter=H['solver']['use_jitter'])
            d = gen.next()
            sess.run(enqueue_op[phase], feed_dict=make_feed(d))
            threads.append(tf.train.threading.Thread(target=my_loop,
                                                     args=(coord, sess, enqueue_op, phase, gen)))
            threads[-1].start()

        tf.set_random_seed(H['solver']['rnd_seed'])
        sess.run(tf.initialize_all_variables())

        weights_str = H['solver']['weights']
        if len(weights_str) > 0:
            print('Restoring from: %s' % weights_str)
            saver.restore(sess, weights_str)

        # train model for N iterations
        start = time.time()
        max_iter = H['solver'].get('max_iter', 10000000)
        for i in xrange(max_iter):
            if coord.should_stop():
                break
            display_iter = H['logging']['display_iter']
            adjusted_lr = (H['solver']['learning_rate'] *
                           0.5 ** max(0, (i / H['solver']['learning_rate_step']) - 2))
            lr_feed = {learning_rate: adjusted_lr}
            if i % display_iter == 0:
                if i > 0:
                    dt = (time.time() - start) / (H['arch']['batch_size'] * display_iter)
                start = time.time()
                (batch_loss_train, test_accuracy, weights_norm,
                    summary_str, np_test_image, np_test_pred_boxes,
                    np_test_pred_confidences, np_test_true_boxes,
                    np_test_true_confidences, _, _) = sess.run([
                         loss['train'], accuracy['test'], W_norm,
                         summary_op, test_image, test_pred_boxes,
                         test_pred_confidences, test_true_boxes, test_true_confidences,
                         train_op, smooth_op,
                        ], feed_dict=lr_feed)
                num_img_logs = 3
                pred_true = [("%d_pred_output" % (i % num_img_logs), np_test_pred_boxes, np_test_pred_confidences),
                             ("%d_true_output" % (i % num_img_logs), np_test_true_boxes, np_test_true_confidences)]

                for name, boxes, confidences in pred_true:
                    test_output_to_log = train_utils.add_rectangles(H,
                                                                    np_test_image,
                                                                    confidences,
                                                                    boxes,
                                                                    H["arch"],
                                                                    use_stitching=True,
                                                                    rnn_len=H['arch']['rnn_len'])[0]
                    assert test_output_to_log.shape == (H['arch']['image_height'],
                                                        H['arch']['image_width'], num_img_logs)
                    feed = {test_image_to_log: test_output_to_log, log_image_name: name}
                    test_image_summary_str = sess.run(log_image, feed_dict=feed)
                    writer.add_summary(test_image_summary_str, global_step=global_step.eval())
                writer.add_summary(summary_str, global_step=global_step.eval())
                print_str = string.join([
                    'Step: %d',
                    'lr: %f',
                    'Train Loss: %.2f',
                    'Test Accuracy: %.1f%%',
                    'Time/image (ms): %.1f'
                ], ', ')
                print(print_str %
                      (i, adjusted_lr, batch_loss_train,
                       test_accuracy * 100, dt * 1000 if i > 0 else 0))
            else:
                batch_loss_train, _ = sess.run([loss['train'], train_op], feed_dict=lr_feed)

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
    H['arch']['num_classes'] = 2
    train(H, test_images=[])

if __name__ == '__main__':
    main()

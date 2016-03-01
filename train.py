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
    lstm_size = H['arch']['lstm_size']
    lstm = rnn_cell.BasicLSTMCell(lstm_size, forget_bias=0.0)
    batch_size = H['arch']['batch_size'] * H['arch']['grid_height'] * H['arch']['grid_width']
    state = tf.zeros([batch_size, lstm.state_size])

    outputs = []
    with tf.variable_scope('RNN'):
        for time_step in range(H['arch']['rnn_len']):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            output, state = lstm(lstm_input, state)
            outputs.append(output)
    return outputs

#inner_dim = 5
#data = np.array([[[[100 * x + i * j + k for k in range(inner_dim)] for j in range(10)] for i in range(10)] for x in range(2)])
#a = tf.constant(data, dtype=tf.float32)
#index = tf.placeholder(tf.float32)
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

def reinspect(H, pred_boxes, early_feat, early_feat_channels):
    grid_size = H['arch']['grid_width'] * H['arch']['grid_height']
    outer_size = grid_size * H['arch']['batch_size']

    fine_stride = 8. # pixels per 60x80 grid cell in 480x640 image
    coarse_stride = 32. # pixels per 15x20 grid cell in 480x640 image
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
    scale_factor = 4 # scale difference between 15x20 and 60x80 features
    pred_x_center = tf.clip_by_value((pred_boxes_r[:, 0:1] + x_offsets) / fine_stride,
    #pred_x_center = tf.clip_by_value((x_offsets) / fine_stride,
                                     0,
                                     scale_factor * H['arch']['grid_width'] - 1)
    pred_y_center = tf.clip_by_value((pred_boxes_r[:, 1:2] + y_offsets) / fine_stride,
    #pred_y_center = tf.clip_by_value((y_offsets) / fine_stride,
                                     0,
                                     scale_factor * H['arch']['grid_height'] - 1)

    #pred_x_center = tf.Print(pred_x_center, [tf.reduce_max(pred_x_center)])
    #pred_y_center = tf.Print(pred_y_center, [tf.reduce_max(pred_y_center)])

    interp_indices = tf.concat(1, [tf.to_float(batch_ids), pred_y_center, pred_x_center])
    reinspect_features = interp(early_feat, interp_indices, early_feat_channels)
    reinspect_features_r = tf.reshape(reinspect_features,
                                      [outer_size, H['arch']['rnn_len'], early_feat_channels])

    return reinspect_features_r

def build_lstm_forward(H, x, googlenet, phase, reuse):
    grid_size = H['arch']['grid_width'] * H['arch']['grid_height']
    outer_size = grid_size * H['arch']['batch_size']
    input_mean = 117.
    x -= input_mean
    global early_feat
    Z, early_feat, early_feat_channels = googlenet_load.model(x, googlenet, H)
    with tf.variable_scope('decoder', reuse=reuse):
        scale_down = 0.01
        if H['arch']['early_dropout'] and phase == 'train':
            Z = tf.nn.dropout(Z, 0.5)
        lstm_input = tf.reshape(Z * scale_down, (H['arch']['batch_size'] * grid_size, 1024))
        lstm_outputs = build_lstm_inner(lstm_input, H)

        pred_boxes = []
        pred_logits = []
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        for k in range(H['arch']['rnn_len']):
            output = lstm_outputs[k]
            if H['arch']['late_dropout'] and phase == 'train':
                output = tf.nn.dropout(output, 0.5)
            box_weights = tf.get_variable('box_ip%d' % k,
                                          shape=(H['arch']['lstm_size'], 4),
                                          initializer=initializer)
            conf_weights = tf.get_variable('conf_ip%d' % k,
                                           shape=(H['arch']['lstm_size'], 2),
                                           initializer=initializer)
            pred_boxes.append(tf.reshape(tf.matmul(output, box_weights) * 50,
                                         [outer_size, 1, 4]))
            pred_logits.append(tf.reshape(tf.matmul(output, conf_weights),
                                         [outer_size, 1, 2]))
 
        pred_boxes = tf.concat(1, pred_boxes)
        pred_logits = tf.concat(1, pred_logits)
        pred_logits_squash = tf.reshape(pred_logits,
                                        [outer_size * H['arch']['rnn_len'], 2])
        pred_confidences_squash = tf.nn.softmax(pred_logits_squash)
        pred_confidences = tf.reshape(pred_confidences_squash,
                                      [outer_size, H['arch']['rnn_len'], 2])

        if H['arch']['use_reinspect']:
            pred_boxes_deltas = []
            reinspect_features = reinspect(H, pred_boxes, early_feat, early_feat_channels)
            for k in range(H['arch']['rnn_len']):
                delta_features = tf.concat(1, [lstm_outputs[k], reinspect_features[:, k, :] / 1000.])
                delta_weights = tf.get_variable(
                                    'delta_ip%d' % k,
                                    shape=[H['arch']['lstm_size'] + early_feat_channels, 4],
                                    initializer=initializer)
                pred_boxes_deltas.append(tf.reshape(tf.matmul(delta_features, delta_weights) * 50,
                                                    [outer_size, 1, 4]))
            pred_boxes_deltas = tf.concat(1, pred_boxes_deltas)
            return pred_boxes, pred_logits, pred_confidences, pred_boxes_deltas

    return pred_boxes, pred_logits, pred_confidences

@ops.RegisterGradient("Hungarian")
def _hungarian_grad(op, *args):
    return map(array_ops.zeros_like, op.inputs)

def build_lstm(H, x, googlenet, phase, boxes, flags):
    grid_size = H['arch']['grid_width'] * H['arch']['grid_height']
    outer_size = grid_size * H['arch']['batch_size']
    reuse = {'train': None, 'test': True}[phase]
    if H['arch']['use_reinspect']:
        (pred_boxes, pred_logits,
         pred_confidences, pred_boxes_deltas) = build_lstm_forward(H, x, googlenet, phase, reuse)
    else:
        pred_boxes, pred_logits, pred_confidences = build_lstm_forward(H, x, googlenet, phase, reuse)
    with tf.variable_scope('decoder', reuse={'train': None, 'test': True}[phase]):
        outer_boxes = tf.reshape(boxes, [outer_size, H['arch']['rnn_len'], 4])
        outer_flags = tf.cast(tf.reshape(flags, [outer_size, H['arch']['rnn_len']]), 'int32')
        assignments, classes, perm_truth, pred_mask = (
            tf.user_ops.hungarian(pred_boxes, outer_boxes, outer_flags))
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
        if H['arch']['use_reinspect']:
            delta_residual = tf.reshape(perm_truth - (pred_boxes + pred_boxes_deltas) * pred_mask,
                                        [outer_size, H['arch']['rnn_len'], 4])
            tf.histogram_summary(phase + '/delta_hist0', pred_boxes_deltas[:, 0, :])
            tf.histogram_summary(phase + '/delta_hist0_w', pred_boxes_deltas[:, 0, 2])
            tf.histogram_summary(phase + '/delta_hist0_x', pred_boxes_deltas[:, 0, 0])
            tf.histogram_summary(phase + '/boxes_hist0', pred_boxes[:, 0, :])
            tf.histogram_summary(phase + '/early_feats', early_feat)
            deltas_loss = (tf.reduce_sum(tf.abs(delta_residual)) / 
                           outer_size * H['solver']['head_weights'][2])
            loss = confidences_loss + boxes_loss + deltas_loss
            # TODO: remove this
            boxes_loss = deltas_loss
            pred_boxes = pred_boxes + pred_boxes_deltas
        else:
            loss = confidences_loss + boxes_loss

    return pred_boxes, pred_confidences, loss, confidences_loss, boxes_loss

def build_overfeat_forward(H, x, googlenet, phase):
    input_mean = 117.
    x -= input_mean
    Z, _, _ = googlenet_load.model(x, googlenet, H)
    grid_size = H['arch']['grid_width'] * H['arch']['grid_height']
    if H['arch']['use_dropout'] and phase == 'train':
        Z = tf.nn.dropout(Z, 0.5)
    pred_logits = tf.reshape(tf.nn.xw_plus_b(Z, googlenet['W'][0], googlenet['B'][0],
                                             name=phase+'/logits_0'),
                             [H['arch']['batch_size'] * grid_size, H['arch']['num_classes']])
    pred_confidences = tf.nn.softmax(pred_logits)
    pred_boxes = tf.reshape(tf.nn.xw_plus_b(Z, googlenet['W'][1], googlenet['B'][1],
                                            name=phase+'/logits_1'),
                            [H['arch']['batch_size'] * grid_size, 1, 4]) * 100
    return pred_boxes, pred_logits, pred_confidences

def build_overfeat(H, x, googlenet, phase, boxes, confidences_r):
    pred_boxes, pred_logits, pred_confidences = build_overfeat_forward(H, x, googlenet, phase)

    grid_size = H['arch']['grid_width'] * H['arch']['grid_height']
    boxes = tf.cast(tf.reshape(boxes, [H['arch']['batch_size'] * grid_size, 4]), 'float32')
    cross_entropy = -tf.reduce_sum(confidences_r*tf.log(tf.nn.softmax(pred_logits) + 1e-6))

    L = (H['solver']['head_weights'][0] * cross_entropy,
         H['solver']['head_weights'][1] * tf.abs(pred_boxes[:, 0, :] - boxes) *
             tf.expand_dims(confidences_r[:, 1], 1))
    confidences_loss = (tf.reduce_sum(L[0], name=phase+'/confidences_loss') /
                        (H['arch']['batch_size'] * grid_size))
    boxes_loss = (tf.reduce_sum(L[1], name=phase+'/boxes_loss') /
                  (H['arch']['batch_size'] * grid_size))
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

    googlenet = googlenet_load.init(H, config)
    learning_rate = tf.placeholder(tf.float32)
    if solver['opt'] == 'RMS':
        opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                        decay=0.9, epsilon=solver['epsilon'])
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
        confidences_r = tf.cast(
            tf.reshape(confidences[:, :, 0, :],
                       [H['arch']['batch_size'] * grid_size, arch['num_classes']]), 'float32')

        if arch['use_lstm']:
            (pred_boxes, pred_confidences,
             loss[phase], confidences_loss[phase],
             boxes_loss[phase]) = build_lstm(H, x, googlenet, phase, boxes, flags)
            pred_confidences = pred_confidences[:, 0, :]
        else:
            (pred_boxes, pred_confidences,
             loss[phase], confidences_loss[phase],
             boxes_loss[phase]) = build_overfeat(H, x, googlenet, phase, boxes, confidences_r)


        # Set up summary operations for tensorboard
        a = tf.equal(tf.argmax(confidences_r, 1), tf.argmax(pred_confidences, 1))
        accuracy[phase] = tf.reduce_mean(tf.cast(a, 'float32'), name=phase+'/accuracy')

        if phase == 'train':
            global_step = tf.Variable(0, trainable=False)
            train_op = opt.minimize(loss['train'], global_step=global_step)
        elif phase == 'test':
            test_image = x
            moving_avg = tf.train.ExponentialMovingAverage(0.99)
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

            # show ground truth to verify labels are correct
            test_true_confidences = confidences_r
            test_true_boxes = boxes[0, :, 0, :]

            # show predictions to visualize training progress
            test_pred_confidences = pred_confidences
            test_pred_boxes = pred_boxes[:, 0, :]

    summary_op = tf.merge_all_summaries()

    return (config, loss, accuracy, summary_op, train_op, googlenet['W_norm'],
            test_image, test_pred_boxes, test_pred_confidences,
            test_true_boxes, test_true_confidences, smooth_op,
            global_step, learning_rate)


def train(H, test_images):
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

    def MyLoop(sess, enqueue_op, phase, gen):
        for d in gen:
            sess.run(enqueue_op[phase], feed_dict=make_feed(d))

    (config, loss, accuracy, summary_op, train_op, W_norm,
     test_image, test_pred_boxes, test_pred_confidences,
     test_true_boxes, test_true_confidences,
     smooth_op, global_step, learning_rate) = build(H, q)

    saver = tf.train.Saver(max_to_keep=None)
    writer = tf.train.SummaryWriter(
        logdir=H['save_dir'],
        flush_secs=10
    )

    test_image_to_log = tf.placeholder(tf.uint8,
                                       [H['arch']['image_height'], H['arch']['image_width'], 3])
    log_image_name = tf.placeholder(tf.string)
    log_image = tf.image_summary(log_image_name, tf.expand_dims(test_image_to_log, 0))

    with tf.Session(config=config) as sess:
        threads = []
        for phase in ['train', 'test']:
            # enqueue once manually to avoid thread start delay
            gen = train_utils.load_data_gen(H, phase, jitter=H['solver']['use_jitter'])
            d = gen.next()
            sess.run(enqueue_op[phase], feed_dict=make_feed(d))
            threads.append(tf.train.threading.Thread(target=MyLoop,
                                                     args=(sess, enqueue_op, phase, gen)))
            threads[-1].start()

        tf.set_random_seed(H['solver']['rnd_seed'])
        sess.run(tf.initialize_all_variables())

        weights_str = H['solver']['weights']
        if len(weights_str) > 0:
            print('Restoring from: %s' % weights_str)
            saver.restore(sess, weights_str)

        # train model for N iterations
        for i in xrange(10000000):
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
                pred_true = [("%d_pred_output" % (i % 3), np_test_pred_boxes, np_test_pred_confidences),
                             ("%d_true_output" % (i % 3), np_test_true_boxes, np_test_true_confidences)]

                for name, boxes, confidences in pred_true:
                    test_output_to_log = train_utils.add_rectangles(np_test_image,
                                                                    confidences,
                                                                    boxes,
                                                                    H["arch"])[0]
                    assert test_output_to_log.shape == (H['arch']['image_height'],
                                                        H['arch']['image_width'], 3)
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

            if global_step.eval() % H['logging']['save_iter'] == 0:
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

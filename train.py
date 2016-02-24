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

def build_lstm_forward(H, x, googlenet, phase, reuse):
    grid_size = H['arch']['grid_width'] * H['arch']['grid_height']
    outer_size = grid_size * H['arch']['batch_size']
    input_mean = 117.
    x -= input_mean
    Z = googlenet_load.model(x, googlenet, H)
    with tf.variable_scope('decoder', reuse=reuse):
        scale_down = 0.01
        if H['arch']['early_dropout'] and phase == 'train':
            Z = tf.nn.dropout(Z, 0.5)
        lstm_input = tf.reshape(Z * scale_down, (H['arch']['batch_size'] * grid_size, 1024))
        lstm_outputs = build_lstm_inner(lstm_input, H)

        pred_boxes = []
        pred_logits = []
        for i in range(H['arch']['rnn_len']):
            output = lstm_outputs[i]
            if H['arch']['late_dropout'] and phase == 'train':
                output = tf.nn.dropout(output, 0.5)
            box_weights = tf.get_variable('box_ip%d' % i, shape=(H['arch']['lstm_size'], 4),
                initializer=tf.random_uniform_initializer(0.1))
            conf_weights = tf.get_variable('conf_ip%d' % i, shape=(H['arch']['lstm_size'], 2),
                initializer=tf.random_uniform_initializer(0.1))
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
    return pred_boxes, pred_logits, pred_confidences

@ops.RegisterGradient("Hungarian")
def _hungarian_grad(op, *args):
    return map(array_ops.zeros_like, op.inputs)

def build_lstm(H, x, googlenet, phase, boxes, box_flags):
    grid_size = H['arch']['grid_width'] * H['arch']['grid_height']
    outer_size = grid_size * H['arch']['batch_size']
    reuse = {'train': None, 'test': True}[phase]
    pred_boxes, pred_logits, pred_confidences = build_lstm_forward(H, x, googlenet, phase, reuse)
    with tf.variable_scope('decoder', reuse={'train': None, 'test': True}[phase]):
        outer_boxes = tf.reshape(boxes, [outer_size, H['arch']['rnn_len'], 4])
        outer_flags = tf.cast(tf.reshape(box_flags, [outer_size, H['arch']['rnn_len']]), 'int32')
        assignments, classes, perm_truth, pred_mask = (
            tf.user_ops.hungarian(pred_boxes, outer_boxes, outer_flags))
        true_classes = tf.reshape(tf.cast(tf.greater(classes, 0), 'int64'),
                                  [outer_size * H['arch']['rnn_len']])
        pred_logit_r = tf.reshape(pred_logits,
                                  [outer_size * H['arch']['rnn_len'], 2])
        confidences_loss = (tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(pred_logit_r, true_classes))
            ) / outer_size * H['solver']['head_weights'][0]
        residual = tf.reshape(pred_boxes * pred_mask - perm_truth,
                              [outer_size, H['arch']['rnn_len'], 4])
        boxes_loss = tf.reduce_sum(tf.abs(residual)) / outer_size * H['solver']['head_weights'][1]
        loss = confidences_loss + boxes_loss
    return pred_boxes, pred_confidences, loss, confidences_loss, boxes_loss

def build_overfeat_forward(H, x, googlenet, phase):
    input_mean = 117.
    x -= input_mean
    Z = googlenet_load.model(x, googlenet, H)
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
        x, confidences, boxes, box_flags = q[phase].dequeue_many(arch['batch_size'])


        grid_size = H['arch']['grid_width'] * H['arch']['grid_height']
        confidences_r = tf.cast(
            tf.reshape(confidences,
                       [H['arch']['batch_size'] * grid_size, arch['num_classes']]), 'float32')

        if arch['use_lstm']:
            (pred_boxes, pred_confidences,
             loss[phase], confidences_loss[phase],
             boxes_loss[phase]) = build_lstm(H, x, googlenet, phase, boxes, box_flags)
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
    flags_in = tf.placeholder(tf.float32)
    q = {}
    enqueue_op = {}
    for phase in ['train', 'test']:
        dtypes = [tf.float32, tf.float32, tf.float32, tf.float32]
        grid_size = H['arch']['grid_width'] * H['arch']['grid_height']
        shapes = (
            [H['arch']['image_height'], H['arch']['image_width'], 3],
            [grid_size, 2],
            [grid_size, H['arch']['rnn_len'], 4],
            [grid_size, H['arch']['rnn_len']],
            )
        q[phase] = tf.FIFOQueue(capacity=30, dtypes=dtypes, shapes=shapes)
        enqueue_op[phase] = q[phase].enqueue((x_in, confs_in, boxes_in, flags_in))

    def make_feed(d):
        return {x_in: d['image'], confs_in: d['confs'], boxes_in: d['boxes'],
                flags_in: d['flags'], learning_rate: H['solver']['learning_rate']}

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
            display_iter = 10
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
                                                                    H["arch"])
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

            if global_step.eval() % 1000 == 0: 
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

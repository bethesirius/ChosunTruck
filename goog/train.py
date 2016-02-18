import json
import datetime
import random
import time
import string
import argparse
import os
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.models.rnn import rnn_cell

random.seed(0)
np.random.seed(0)

import train_utils
from utils import add_rectangles
import googlenet_load


@ops.RegisterGradient("Hungarian")
def _hungarian_grad(op, *args):
    return map(array_ops.zeros_like, op.inputs)

def build_lstm(lstm_input, H):
    lstm_size = H['net']['lstm_size']
    lstm = rnn_cell.BasicLSTMCell(lstm_size, forget_bias=0.0)
    batch_size = H['net']['batch_size'] * H['net']['grid_height'] * H['net']['grid_width']
    state = tf.zeros([batch_size, lstm.state_size])

    ip = tf.get_variable('ip', [lstm_size, 1], initializer=tf.random_uniform_initializer(0.1))
    rnn_len = H['net']['rnn_len']

    outputs = []
    with tf.variable_scope('RNN'):
        for time_step in range(H['net']['rnn_len']):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            output, state = lstm(lstm_input, state)
            outputs.append(output)
    return outputs

def build(H, q):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(H['gpu'])
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    gpu_options = tf.GPUOptions()
    config = tf.ConfigProto(gpu_options=gpu_options)
    net_config = H['net']

    k = 2
    W, B, weight_tensors, reuse_ops, input_op, W_norm = googlenet_load.setup(config, k)
    num_threads = 6
    input_mean = 117.
    #opt = tf.train.RMSPropOptimizer(learning_rate=H['learning_rate'], decay=0.9, epsilon=H['epsilon'])
    opt = tf.train.GradientDescentOptimizer(learning_rate=H['learning_rate'])
    files, loss, accuracy = {}, {}, {}
    for phase in ['train', 'test']:
        x, confidences, boxes, box_flags = q[phase].dequeue_many(net_config['batch_size'])

        if phase == 'train':
            y_density = tf.cast(tf.argmax(confidences, 2), 'uint8')
            y_density_image = tf.reshape(y_density[0, :], [1, H['net']['grid_height'], H['net']['grid_width'], 1])

            tf.image_summary('train_label', y_density_image)
            tf.image_summary('train_image', x[0:1, :, :, :])

        x -= input_mean

        if phase == 'test':
            test_boxes = boxes
            test_image = x + input_mean
            test_confidences = confidences

        Z = googlenet_load.model(x, weight_tensors, input_op, reuse_ops, H)

        if H['use_dropout'] and phase == 'train':
            Z = tf.nn.dropout(Z, 0.5)
        grid_size = H['net']['grid_width'] * H['net']['grid_height']
        scale_down = 0.01
        lstm_input = tf.reshape(Z * scale_down, (H['net']['batch_size'] * grid_size, 1024))
        with tf.variable_scope('decoder', reuse={'train': None, 'test': True}[phase]):
            lstm_outputs = build_lstm(lstm_input, H)
            #pred = tf.matmul(output, ip)

            pred_boxes = []
            pred_confs = []
            for i in range(H['net']['rnn_len']):
                output = lstm_outputs[i]
                box_weights = tf.get_variable('box_ip%d' % i, shape=(H['net']['lstm_size'], 4),
                    initializer=tf.random_uniform_initializer(0.1))
                conf_weights = tf.get_variable('conf_ip%d' % i, shape=(H['net']['lstm_size'], 2),
                    initializer=tf.random_uniform_initializer(0.1))
                pred_boxes.append(tf.matmul(output, box_weights) * 50)
                pred_confs.append(tf.matmul(output, conf_weights))
            #assignments, classes, perm_truth, pred_mask = tf.user_ops.hungarian(pred_boxes, boxes, tf.cast(box_flags, 'int32'))
            #tf.user_ops.hungarian(pred_boxes, boxes, tf.cast(box_flags, 'int32'))

                #loss = tf.reduce_sum(tf.pow(tf.reduce_sum(value, 1) - pred[:, 0], 2))
                #lr = tf.placeholder(tf.float32)
                #opt = tf.train.GradientDescentOptimizer(lr)
                #train_op = opt.minimize(loss)

        pred_logits = tf.reshape(tf.nn.xw_plus_b(Z, W[0], B[0], name=phase+'/logits_0'), 
              [H['net']['batch_size'] * grid_size, k])
        pred_confidences = tf.nn.softmax(pred_logits)

        pred_boxes = tf.reshape(tf.nn.xw_plus_b(Z, W[1], B[1], name=phase+'/logits_1'), 
              [H['net']['batch_size'] * grid_size, 4]) * 100

        confidences = tf.cast(tf.reshape(confidences, [H['net']['batch_size'] * grid_size, k]), 'float32')
        boxes = tf.cast(tf.reshape(boxes, [H['net']['batch_size'] * grid_size, 4]), 'float32')
        if phase == 'test':
            test_pred_confidences = pred_confidences
            test_pred_confidences = confidences
            test_pred_boxes = pred_boxes


        if phase == 'train':
            sm = tf.nn.softmax(pred_logits)
            #pred_logits = tf.Print(pred_logits, [pred_logits, sm], "Pred_logits: ", summarize=30000)
            #confidences = tf.Print(confidences, [confidences], "Confidences: ", summarize=30000)
            cross_entropy = -tf.reduce_sum(confidences*tf.log(tf.nn.softmax(pred_logits) + 1e-6))
            #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(pred_logits, confidences)
            #cross_entropy = tf.Print(cross_entropy, [cross_entropy], "Cross-entropy: ", summarize=3000)


            L = (H['head_weights'][0] * cross_entropy,
            #L = (H['head_weights'][0] * 0,
                 #)
                 H['head_weights'][1] * tf.abs(pred_boxes - boxes) * tf.expand_dims(confidences[:, 1], 1))

            confidences_loss = tf.reduce_sum(L[0], name=phase+'/confidences_loss') / (H['net']['batch_size'] * grid_size)
            boxes_loss = tf.reduce_sum(L[1], name=phase+'/boxes_loss') / (H['net']['batch_size'] * grid_size)
            tf.scalar_summary("%s/confidences_loss" % phase, confidences_loss)
            tf.scalar_summary("%s/regression_loss" % phase, boxes_loss)

            L = 0.
            if H['head_weights'][0] > 0.:
                print('using confidences loss')
                L += confidences_loss
            if H['head_weights'][1] > 0.:
                print('using box loss')
                L += boxes_loss

            loss[phase] = L

        a = tf.equal(tf.argmax(confidences, 1), tf.argmax(pred_confidences, 1))
        a = tf.reduce_mean(tf.cast(a, 'float32'), name=phase+'/accuracy')
        tf.scalar_summary(a.op.name, a)
        accuracy[phase] = a

        if phase == 'test':
            moving_avg = tf.train.ExponentialMovingAverage(0.95)
            smooth_op = moving_avg.apply([a])
            tf.scalar_summary(a.op.name + '/smooth', moving_avg.average(a))

        if phase == 'train':
            global_step = tf.Variable(0, trainable=False)
            train_op = opt.minimize(L, global_step=global_step)

    summary_op = tf.merge_all_summaries()

    return config, loss, accuracy, W_norm, summary_op, train_op, test_image, test_boxes, test_confidences, test_pred_boxes, test_pred_confidences, smooth_op, global_step

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=None, type=str)
    parser.add_argument('--load_fast', action='store_true')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        H = json.load(f)
    assert 'gpu' not in H
    H['gpu'] = args.gpu
    H['save_dir'] = 'log/%s_%s' % (H['logging']['logdir'],
        datetime.datetime.now().strftime('%Y_%m_%d_%H.%M'))
    if not os.path.exists(H['save_dir']): os.makedirs(H['save_dir'])

    ckpt_file = H['save_dir'] + '/save.ckpt'
    with open(H['save_dir'] + '/hypes.json', 'w') as f:
        json.dump(H, f, indent=4)
    net_config = H['net']

    x_in = tf.placeholder(tf.float32)
    confs_in = tf.placeholder(tf.float32)
    boxes_in = tf.placeholder(tf.float32)
    flags_in = tf.placeholder(tf.float32)
    q = {}
    enqueue_op = {}
    for phase in ['train', 'test']:
        dtypes = [tf.float32, tf.float32, tf.float32, tf.float32]
        grid_size = net_config['grid_width'] * net_config['grid_height']
        shapes = (
            [net_config['image_height'], net_config['image_width'], 3],
            [grid_size, 2],
            [grid_size, net_config['rnn_len'], 4],
            [grid_size, net_config['rnn_len']],
            )
        q[phase] = tf.FIFOQueue(capacity=30, dtypes=dtypes, shapes=shapes)
        enqueue_op[phase] = q[phase].enqueue((x_in, confs_in, boxes_in, flags_in))

    def make_feed(d):
        return {x_in: d['image'], confs_in: d['confs'], boxes_in: d['boxes'], flags_in: d['flags']}

    def MyLoop(sess, enqueue_op, phase, gen):
        for d in gen:
            sess.run(enqueue_op[phase], feed_dict=make_feed(d))
    val_or_test = 'val'

    config, loss, accuracy, W_norm, summary_op, train_op, test_image, test_boxes, test_confidences, test_pred_boxes, test_pred_confidences, smooth_op, global_step = build(H, q)
    #check_op = tf.add_check_numerics_ops()

    saver = tf.train.Saver(max_to_keep=None)
    writer = tf.train.SummaryWriter(
        logdir=H['save_dir'], 
        flush_secs=10
    )
    writer.add_graph(tf.get_default_graph().as_graph_def())

    test_image_to_log = tf.placeholder(tf.uint8, [net_config['image_height'], net_config['image_width'], 3])
    log_image = tf.image_summary('test_image_to_log', tf.expand_dims(test_image_to_log, 0))


    with tf.Session(config=config) as sess:
        # enqueue once manually to avoid thread start delay
        for phase in ['train', 'test']:
            gen = train_utils.load_data_gen(H, phase, val_or_test)
            d = gen.next()
            sess.run(enqueue_op[phase], feed_dict=make_feed(d))
            thread = tf.train.threading.Thread(target=MyLoop, args=(sess, enqueue_op, phase, gen))
            thread.start()

        tf.set_random_seed(0)
        sess.run(tf.initialize_all_variables())

        if args.weights is not None:
            saver.restore(sess, args.weights)

        for i in xrange(10000000):
            try:
                display_iter = 10
                if i % display_iter == 0:
                    if i > 0:
                        dt = (time.time() - start) / (H['net']['batch_size'] * display_iter)
                    start = time.time()
                    (batch_loss_train, test_accuracy, weights_norm, 
                        summary_str, np_test_image, np_test_boxes, np_test_confidences,
                        _, _) = sess.run([loss['train'], accuracy['test'], W_norm, 
                            summary_op, test_image, test_pred_boxes, test_pred_confidences, train_op, smooth_op])
                    test_output_to_log = add_rectangles(np_test_image, np_test_confidences, np_test_boxes, H["net"])
                    feed = {test_image_to_log: test_output_to_log}
                    test_image_summary_str = sess.run(log_image, feed_dict=feed)
                    assert test_output_to_log.shape == (net_config['image_height'], net_config['image_width'], 3)
                    writer.add_summary(test_image_summary_str, global_step=global_step.eval())
                    writer.add_summary(summary_str, global_step=global_step.eval())
                    print_str = string.join([
                        'Step: %d',
                        'Train Loss: %.2f',
                        'Test Accuracy: %.1f%%',
                        'Time/image (ms): %.1f'
                    ], ', ')
                    print(print_str % (
                        i,
                        batch_loss_train,
                        test_accuracy * 100,
                        dt * 1000 if i > 0 else 0
                    ))
                else:
                    (batch_loss_train,
                        _) = sess.run([loss['train'],
                            train_op])

                if global_step.eval() % 1000 == 0: 
                    saver.save(sess, ckpt_file, global_step=global_step)
            except Exception as ex:
                import ipdb; ipdb.set_trace()



if __name__ == '__main__':
    main()

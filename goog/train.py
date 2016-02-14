import json
import random
import time
import string
import argparse
import os
import tensorflow as tf
import numpy as np

random.seed(0)
np.random.seed(0)


import train_utils
from utils import add_rectangles
import googlenet_load

def build(H):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(H['gpu'])
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    gpu_options = tf.GPUOptions()
    config = tf.ConfigProto(gpu_options=gpu_options)

    k = 10
    W, B, weight_tensors, reuse_ops, input_op, W_norm = googlenet_load.setup(config, k)
    data = train_utils.load_data(H['load_fast'])
    num_threads = 6
    input_mean = 117.
    opt = tf.train.RMSPropOptimizer(learning_rate=H['learning_rate'], decay=0.9, epsilon=H['epsilon'])
    files, loss, accuracy = {}, {}, {}
    for phase in ['train', 'test']:
        X = data[phase]['X']
        Y = data[phase]['Y']
        boxes = data[phase]['boxes']

        files[phase], confidences, boxes = tf.train.slice_input_producer(
            tensor_list=[X, Y, boxes],
            shuffle=True,
            capacity=H['net']['batch_size']*(num_threads+1)
        )

        x = tf.read_file(files[phase])
        x = tf.image.decode_png(x, channels=3)
        x = tf.image.resize_images(x, H['image_height'], H['image_width'])
        if phase == 'train':
            y_density = tf.cast(tf.argmax(confidences, 1) * 128, 'uint8')
            y_density_image = tf.reshape(y_density, [1, 15, 20, 1])
            tf.image_summary('train_label', y_density_image)
            tf.image_summary('train_image', tf.expand_dims(x, 0))

        x -= input_mean

        x, confidences, boxes, files[phase] = tf.train.shuffle_batch(
            tensor_list = [x, confidences, boxes, files[phase]],
            batch_size = H['net']['batch_size'],
            num_threads = num_threads,
            capacity = H['net']['batch_size']*(num_threads+1),
            min_after_dequeue = H['net']['batch_size']
        )
        if phase == 'test':
            test_boxes = boxes
            test_image = x + input_mean
            test_confidences = confidences

        Z = googlenet_load.model(x, weight_tensors, input_op, reuse_ops, H)

        if H['use_dropout'] and phase == 'train':
            Z = tf.nn.dropout(Z, 0.5)

        pred_logits = tf.reshape(tf.nn.xw_plus_b(Z, W[0], B[0], name=phase+'/logits_0'), 
              [H['net']['batch_size'] * 300, k])
        pred_confidences = tf.nn.softmax(pred_logits)

        pred_boxes = tf.reshape(tf.nn.xw_plus_b(Z, W[1], B[1], name=phase+'/logits_1'), 
              [H['net']['batch_size'] * 300, 4]) * 100

        confidences = tf.reshape(confidences, [H['net']['batch_size'] * 300, k])
        boxes = tf.cast(tf.reshape(boxes, [H['net']['batch_size'] * 300, 4]), 'float32')
        if phase == 'test':
            test_pred_confidences = pred_confidences
            test_pred_boxes = pred_boxes

        cross_entropy = -tf.reduce_sum(confidences*tf.log(pred_confidences + 1e-7))

        L = (H['head_weights'][0] * cross_entropy,
             H['head_weights'][1] * tf.abs(pred_boxes - boxes) * tf.expand_dims(confidences[:, 1], 1))

        confidences_loss = tf.reduce_sum(L[0], name=phase+'/confidences_loss') / (H['net']['batch_size'] * 300)
        boxes_loss = tf.reduce_sum(L[1], name=phase+'/boxes_loss') / (H['net']['batch_size'] * 300)
        tf.scalar_summary(confidences_loss.op.name, confidences_loss)
        tf.scalar_summary(boxes_loss.op.name, boxes_loss)

        L = 0.
        if H['head_weights'][0] > 0.:
            print('using confidences loss')
            L += confidences_loss
        if H['head_weights'][1] > 0.:
            print('using box loss')
            L += boxes_loss

        #L = boxes_loss + confidences_loss
        #L = confidences_loss
        loss[phase] = L

        a = tf.equal(tf.argmax(confidences, 1), tf.argmax(pred_confidences, 1))
        a = tf.reduce_mean(tf.cast(a, 'float32'), name=phase+'/accuracy')
        tf.scalar_summary(a.op.name, a)
        accuracy[phase] = a

        if phase == 'test':
            moving_avg = tf.train.ExponentialMovingAverage(0.95)
            smooth_op = moving_avg.apply([a])
            tf.scalar_summary(a.op.name + '/smooth', moving_avg.average(a))

            #y_out0 = tf.reshape(pred_confidences, [H['net']['batch_size'], 300, k])[0, :, :]
            #y_density = tf.cast(y_out0[:, 1] * 128, 'uint8')
            #y_density_image = tf.reshape(y_density, [1, 15, 20, 1])
            #tf.image_summary('test_output', y_density_image)
            ##x = tf.image.resize_images(x, H['image_height'], H['image_width'])
            ##y_density = tf.cast(tf.argmax(y_out0, 1) * 128, 'uint8')
            ###x = tf.image.random_crop(x, [H['image_height'], H['image_width']])
            ###x = tf.image.random_flip_left_right(x)
            ###x = tf.image.random_flip_up_down(x)
            ##tf.image_summary('train_image', tf.expand_dims(x, 0))

        if phase == 'train':
            #grads = opt.compute_gradients(L)
            global_step = tf.Variable(0, trainable=False)
            #train_op = opt.apply_gradients(grads, global_step=global_step)
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
    assert 'load_fast' not in H
    H['load_fast'] = args.load_fast
    H['save_dir'] = 'log/default' + str(time.time())
    if not os.path.exists(H['save_dir']): os.makedirs(H['save_dir'])

    ckpt_file = H['save_dir'] + '/save.ckpt'
    with open(H['save_dir'] + '/hypes.json', 'w') as f:
        json.dump(H, f, indent=4)

    config, loss, accuracy, W_norm, summary_op, train_op, test_image, test_boxes, test_confidences, test_pred_boxes, test_pred_confidences, smooth_op, global_step = build(H)
    check_op = tf.add_check_numerics_ops()


    coord = tf.train.Coordinator()
    saver = tf.train.Saver(max_to_keep=None)
    writer = tf.train.SummaryWriter(
        logdir=H['save_dir'], 
        flush_secs=10
    )
    writer.add_graph(tf.get_default_graph().as_graph_def())

    test_image_to_log = tf.placeholder(tf.uint8, [480, 640, 3])
    log_image = tf.image_summary('test_image_to_log', tf.expand_dims(test_image_to_log, 0))

    with tf.Session(config=config) as sess:

        tf.set_random_seed(0)
        sess.run(tf.initialize_all_variables())
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if args.weights is not None:
            saver.restore(sess, args.weights)

        while not coord.should_stop():
            try:
                t = time.time()

                (batch_loss_train, test_accuracy, weights_norm, 
                    summary_str, np_test_image, np_test_boxes, np_test_confidences,
                    _, _, _) = sess.run([loss['train'], accuracy['test'], W_norm, 
                        summary_op, test_image, test_pred_boxes, test_pred_confidences, train_op, smooth_op, check_op])

                test_output_to_log = add_rectangles(np_test_image, np_test_confidences, np_test_boxes, H["net"])
                feed = {test_image_to_log: test_output_to_log}
                sess.run(log_image, feed_dict=feed)

                assert test_output_to_log.shape == (480, 640, 3)


                dt = (time.time() - t) / H['net']['batch_size']

                writer.add_summary(summary_str, global_step=global_step.eval())
                print_str = string.join([
                    'Step: %d',
                    'Train Loss: %.2f',
                    'Test Accuracy: %.1f%%',
                    'W_norm/Loss: %.2f',
                    'Time/image (ms): %.1f'
                ], ', ')
                print(print_str % (
                    global_step.eval(),
                    batch_loss_train,
                    test_accuracy * 100,
                    weights_norm, dt * 1000
                ))

                if global_step.eval() % 1000 == 0: 
                    saver.save(sess, ckpt_file, global_step=global_step)
            except:
                import ipdb; ipdb.set_trace()


    coord.join(threads)

if __name__ == '__main__':
    main()

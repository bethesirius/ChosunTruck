import tensorflow as tf
import numpy as np
from numpy import random as rnd
import os
import train_utils
import time
import string
import json
import argparse
import cv2
from utils import Rect, stitch_rects


def add_rectangles(orig_image, confidences, boxes, net_config):
    image = np.copy(orig_image[0])
    num_cells = net_config["grid_height"] * net_config["grid_width"]
    num_rects_per_cell = 1
    boxes_r = np.reshape(boxes, (net_config["batch_size"],
                                 net_config["grid_height"],
                                 net_config["grid_width"],
                                 num_rects_per_cell,
                                 4))
    confidences_r = np.reshape(confidences, (net_config["batch_size"],
                                             net_config["grid_height"],
                                             net_config["grid_width"],
                                             num_rects_per_cell,
                                             10))
                                             
    cell_pix_size = 32
    all_rects = [[[] for _ in range(net_config["grid_width"])] for _ in range(net_config["grid_height"])]
    for n in range(num_rects_per_cell):
        for y in range(net_config["grid_height"]):
            for x in range(net_config["grid_width"]):
                bbox = boxes_r[0, y, x, n, :]
                conf = confidences_r[0, y, x, n, 1]
                abs_cx = int(bbox[0]) + cell_pix_size/2 + cell_pix_size * x
                abs_cy = int(bbox[1]) + cell_pix_size/2 + cell_pix_size * y
                w = bbox[2]
                h = bbox[3]
                all_rects[y][x].append(Rect(abs_cx,abs_cy,w,h,conf))

    #print(confidences_r[0,:,:,0,1])

    acc_rects = [r for row in all_rects for cell in row for r in cell]

    for rect in acc_rects:
        if rect.confidence > 0.5:
            cv2.rectangle(image, 
                (rect.cx-int(rect.width/2), rect.cy-int(rect.height/2)), 
                (rect.cx+int(rect.width/2), rect.cy+int(rect.height/2)), 
                (255,0,0),
                2)

    return image

def model(x, weights_dict, input_op, reuse_ops, H):
    def is_early_loss(name):
        early_loss_layers = ['head0', 'nn0', 'softmax0', 'head1', 'nn1', 'softmax1', 'output1']
        return any(name.startswith(prefix) for prefix in early_loss_layers)

    T = weights_dict
    T[input_op.name] = x

    for op in reuse_ops:

        if op.type == 'BatchNormWithGlobalNormalization':

            t, mu, sig, beta, gamma = [T[t.op.name] for t in list(op.inputs)]
            mu, sig = tf.nn.moments(t, [0,1,2])
            T[op.name] = tf.nn.batch_norm_with_global_normalization(
                t=t, m=mu, v=sig, beta=beta, gamma=gamma, 
                variance_epsilon=1e-8, scale_after_normalization=True, 
                name=op.name
            )

        else:

            if is_early_loss(op.name):
                continue
            elif op.name == 'avgpool0':
                pool_op = tf.nn.avg_pool(T['mixed5b'], ksize=[1,15,20,1], strides=[1,1,1,1], padding='VALID', name=op.name)
                T[op.name] = pool_op

            else:
                copied_op = x.graph.create_op(
                    op_type = op.type, 
                    inputs = [T[t.op.name] for t in list(op.inputs)], 
                    dtypes = [o.dtype for o in op.outputs], 
                    name = op.name, 
                    attrs =  op.node_def.attr
                )

                T[op.name] = copied_op.outputs[0]
    

    cnn_feat = T['mixed5b']
    cnn_feat_r = tf.reshape(cnn_feat, [H['net']['batch_size'] * 15 * 20, 1024])

    Z = cnn_feat_r

    return Z

def build(H):

    gpu_id = 0
    num_threads = 6

    features_dim = 1024
    input_mean = 117.
    input_layer = 'input'

    #features_layers = ['nn0/reshape', 'nn1/reshape', 'avgpool0/reshape']
    #head_weights = [0.2, 0.2, 0.6]

    features_layers = ['output/confidences', 'output/boxes']
    head_weights = [1.0, 0.1]

    opt = tf.train.AdamOptimizer(learning_rate=H['learning_rate'], epsilon=H['epsilon'])
    graph_def_orig_file = 'graphs/{}.pb'.format(H['base_net'])

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    config = tf.ConfigProto(gpu_options=gpu_options)

    data = train_utils.load_data(H['load_fast'])

    k = len(data['train']['Y'][0][0])

    graphs = {name: tf.Graph() for name in ['orig']}
    graph_def = tf.GraphDef()
    with open(graph_def_orig_file) as f:
        graph_def.MergeFromString(f.read())

    with graphs['orig'].as_default():
        tf.import_graph_def(graph_def, name='')

    input_op = graphs['orig'].get_operation_by_name(input_layer)

    weights_ops = [
        op for op in graphs['orig'].get_operations() 
        if any(x in op.name for x in ['params', 'batchnorm/beta'])
        or any(op.name.endswith(x) for x in [ '_w', '_b'])
        and op.type == 'Const'
    ]

    reuse_ops = [
        op for op in graphs['orig'].get_operations() 
        if op not in weights_ops + [input_op]
        and op.name != 'output'
    ]

    with tf.Session(graph=graphs['orig'], config=config):
        weights_orig = {
            op.name: op.outputs[0].eval()
            for op in weights_ops
        }

    def weight_init(num_output):
        return H['weight_init_size'] * rnd.randn(features_dim, num_output).astype(np.float32)

    def bias_init(num_output):
        return H['weight_init_size'] * rnd.randn(num_output).astype(np.float32)

    dense_layer_num_output = [k, 4]

    W = [
        tf.Variable(weight_init(dense_layer_num_output[i]), 
                    name='softmax/weights_{}'.format(i)) 
        for i in range(len(features_layers))
    ]

    B = [
        tf.Variable(bias_init(dense_layer_num_output[i]),
                    name='softmax/biases_{}'.format(i)) 
        for i in range(len(features_layers))
    ]

    weight_vars = {
        name: tf.Variable(weight, name=name)
        for name, weight in weights_orig.iteritems()
    }

    weight_tensors = {
        name: tf.convert_to_tensor(weight)
        for name, weight in weight_vars.iteritems()
    }

    W_norm = [tf.nn.l2_loss(weight) for weight in weight_vars.values() + W]
    W_norm = tf.reduce_sum(tf.pack(W_norm), name='weights_norm')
    tf.scalar_summary(W_norm.op.name, W_norm)

    files, loss, accuracy = {}, {}, {}

    test_image_to_log = tf.placeholder(tf.uint8, [480, 640, 3])
    tf.image_summary('test_image_to_log', tf.expand_dims(test_image_to_log, 0))

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
            #x = tf.image.random_crop(x, [H['image_height'], H['image_width']])
            #x = tf.image.random_flip_left_right(x)
            #x = tf.image.random_flip_up_down(x)
            tf.image_summary('train_image', tf.expand_dims(x, 0))
        #else:
            #test_image = x

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

        Z = model(x, weight_tensors, input_op, reuse_ops, H)

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

        cross_entropy = -tf.reduce_sum(confidences*tf.log(tf.clip_by_value(pred_confidences,1e-10,1.0)))

        L = (head_weights[0] * cross_entropy,
             head_weights[1] * tf.abs(pred_boxes - boxes) * tf.expand_dims(confidences[:, 1], 1))

        confidences_loss = tf.reduce_sum(L[0], name=phase+'/confidences_loss') / (H['net']['batch_size'] * 300)
        boxes_loss = tf.reduce_sum(L[1], name=phase+'/boxes_loss') / (H['net']['batch_size'] * 300)
        tf.scalar_summary(confidences_loss.op.name, confidences_loss)
        tf.scalar_summary(boxes_loss.op.name, boxes_loss)

        L = boxes_loss + confidences_loss
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

            #x = tf.image.resize_images(x, H['image_height'], H['image_width'])
            y_out0 = tf.reshape(pred_confidences, [H['net']['batch_size'], 300, k])[0, :, :]
            #y_density = tf.cast(tf.argmax(y_out0, 1) * 128, 'uint8')
            y_density = tf.cast(y_out0[:, 1] * 128, 'uint8')
            y_density_image = tf.reshape(y_density, [1, 15, 20, 1])
            tf.image_summary('test_output', y_density_image)
            ##x = tf.image.random_crop(x, [H['image_height'], H['image_width']])
            ##x = tf.image.random_flip_left_right(x)
            ##x = tf.image.random_flip_up_down(x)
            #tf.image_summary('train_image', tf.expand_dims(x, 0))

        if phase == 'train':
            grads = opt.compute_gradients(L + H['weight_decay']*W_norm)
            global_step = tf.Variable(0, trainable=False)
            train_op = opt.apply_gradients(grads, global_step=global_step)

    summary_op = tf.merge_all_summaries()

    return config, loss, accuracy, W_norm, summary_op, train_op, test_image, test_boxes, test_confidences, test_pred_boxes, test_pred_confidences, smooth_op, global_step, test_image_to_log

H = {
    'base_net': 'googlenet',
    'weight_init_size': 1e-3,
    'learning_rate': 1e-3,
    'epsilon': 1.0,
    'weight_decay': 1e-4,
    'use_dropout': True,
    'min_skin_prob': 0.4,
    'min_tax_score': 0.8,
    'save_dir': 'log/default' + str(time.time()),
    'image_width': 640,
    'image_height': 480,
    "net": {
        "grid_height": 15,
        "grid_width": 20,
        "batch_size": 10,
    },
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=None, type=str)
    parser.add_argument('--load_fast', action='store_true')
    args = parser.parse_args()
    assert 'load_fast' not in H
    H['load_fast'] = args.load_fast
    if not os.path.exists(H['save_dir']): os.makedirs(H['save_dir'])

    ckpt_file = H['save_dir'] + '/save.ckpt'
    with open(H['save_dir'] + '/hypes.json', 'w') as f:
        json.dump(H, f, indent=4)

    config, loss, accuracy, W_norm, summary_op, train_op, test_image, test_boxes, test_confidences, test_pred_boxes, test_pred_confidences, smooth_op, global_step, test_image_to_log = build(H)
    check_op = tf.add_check_numerics_ops()


    coord = tf.train.Coordinator()
    saver = tf.train.Saver(max_to_keep=None)
    writer = tf.train.SummaryWriter(
        logdir=H['save_dir'], 
        flush_secs=10
    )
    writer.add_graph(tf.get_default_graph().as_graph_def())

    with tf.Session(config=config) as sess:

        sess.run(tf.initialize_all_variables())
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if args.weights is not None:
            saver.restore(sess, args.weights)

        test_output_to_log = np.ones((H['image_height'], H['image_width'], 3)) * 128
        while not coord.should_stop():
            try:
                t = time.time()

                feed = {test_image_to_log: test_output_to_log}
                (batch_loss_train, test_accuracy, weights_norm, 
                    summary_str, np_test_image, np_test_boxes, np_test_confidences,
                    _, _, _) = sess.run([loss['train'], accuracy['test'], W_norm, 
                        summary_op, test_image, test_pred_boxes, test_pred_confidences, train_op, smooth_op, check_op], feed_dict=feed)

                #print(np_test_boxes.shape)
                test_output_to_log = add_rectangles(np_test_image, np_test_confidences, np_test_boxes, H["net"])
                assert test_output_to_log.shape == (480, 640, 3)


                dt = (time.time() - t) / H['net']['batch_size']

                writer.add_summary(summary_str, global_step=global_step.eval())
                writer.flush()
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
                    H['weight_decay'] * weights_norm, dt * 1000
                ))

                if global_step.eval() % 1000 == 0: 
                    saver.save(sess, ckpt_file, global_step=global_step)
            except:
                import ipdb; ipdb.set_trace()


    coord.join(threads)

if __name__ == '__main__':
    main()

import tensorflow as tf
import os
import json
import subprocess
from scipy.misc import imread, imresize
from scipy import misc

from train import build_lstm_forward, build_overfeat_forward
from utils import googlenet_load
from utils.annolist import AnnotationLib as al
from utils.train_utils import add_rectangles, rescale_boxes

import cv2
import argparse


def add_rectangles(H, orig_image, confidences, boxes, arch, use_stitching=False, rnn_len=1, min_conf=0.5):
    from utils.rect import Rect
    from utils.stitch_wrapper import stitch_rects
    import numpy as np
    image = np.copy(orig_image[0])
    boxes_r = np.reshape(boxes, (-1,
                                 arch["grid_height"],
                                 arch["grid_width"],
                                 rnn_len,
                                 4))
    confidences_r = np.reshape(confidences, (-1,
                                             arch["grid_height"],
                                             arch["grid_width"],
                                             rnn_len,
                                             2))
    cell_pix_size = H['arch']['region_size']
    all_rects = [[[] for _ in range(arch["grid_width"])] for _ in range(arch["grid_height"])]
    for n in range(0, 5):
        for y in range(arch["grid_height"]):
            for x in range(arch["grid_width"]):
                bbox = boxes_r[0, y, x, n, :]
                conf = confidences_r[0, y, x, n, 1]
                abs_cx = int(bbox[0]) + cell_pix_size/2 + cell_pix_size * x
                abs_cy = int(bbox[1]) + cell_pix_size/2 + cell_pix_size * y
                h = bbox[3]
                w = bbox[2]
                #w = h * 0.4
                all_rects[y][x].append(Rect(abs_cx,abs_cy,w,h,conf))

    if use_stitching:
        acc_rects = stitch_rects(all_rects)
    else:
        acc_rects = [r for row in all_rects for cell in row for r in cell if r.confidence > 0.1]


    for rect in acc_rects:
        if rect.confidence > min_conf:
            cv2.rectangle(image,
                (rect.cx-int(rect.width/2), rect.cy-int(rect.height/2)),
                (rect.cx+int(rect.width/2), rect.cy+int(rect.height/2)),
                (0,255,0),
                2)

    rects = []
    for rect in acc_rects:
        r = al.AnnoRect()
        r.x1 = rect.cx - rect.width/2.
        r.x2 = rect.cx + rect.width/2.
        r.y1 = rect.cy - rect.height/2.
        r.y2 = rect.cy + rect.height/2.
        r.score = rect.true_confidence
        rects.append(r)

    return image, rects

def get_image_dir(args):
    weights_iteration = int(args.weights.split('-')[-1])
    image_dir = '%s/images-%d' % (os.path.dirname(args.weights), weights_iteration)
    return image_dir

def get_results(args, H):
    tf.reset_default_graph()
    googlenet = googlenet_load.init(H)
    x_in = tf.placeholder(tf.float32, name='x_in')
    if H['arch']['use_lstm']:
        if H['arch']['use_reinspect']:
            pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas = build_lstm_forward(H, tf.expand_dims(x_in, 0), googlenet, 'test', reuse=None)
            grid_area = H['arch']['grid_height'] * H['arch']['grid_width']
            pred_confidences = tf.reshape(tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * 5, 2])), [grid_area, 5, 2])
            if H['arch']['reregress']:
                pred_boxes = pred_boxes + pred_boxes_deltas
        else:
            pred_boxes, pred_logits, pred_confidences = build_lstm_forward(H, tf.expand_dims(x_in, 0), googlenet, 'test', reuse=None)
    else:
        pred_boxes, pred_logits, pred_confidences = build_overfeat_forward(H, tf.expand_dims(x_in, 0), googlenet, 'test')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, args.weights)

        pred_annolist = al.AnnoList()

        true_annolist = al.parse(args.test_idl)
        data_dir = os.path.dirname(args.test_idl)
        image_dir = get_image_dir(args)
        subprocess.call('mkdir -p %s' % image_dir, shell=True)
        for i in range(len(true_annolist)):
            true_anno = true_annolist[i]
            I = imread('%s/%s' % (data_dir, true_anno.imageName))[:,:,:3]
            if I.shape[0] != H["arch"]["image_height"] or I.shape[1] != H["arch"]["image_width"]:
                true_anno = rescale_boxes(I.shape, true_anno, H["arch"]["image_height"], H["arch"]["image_width"])
                I = imresize(I, (H["arch"]["image_height"], H["arch"]["image_width"]), interp='cubic')
            feed = {x_in: I}
            (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict=feed)
            pred_anno = al.Annotation()
            pred_anno.imageName = true_anno.imageName
            new_img, rects = add_rectangles(H, [I], np_pred_confidences, np_pred_boxes,
                                            H["arch"], use_stitching=True, rnn_len=H['arch']['rnn_len'], min_conf=0.05)
        
            pred_anno.rects = rects
            pred_anno.imagePath = os.path.abspath(data_dir)
            pred_annolist.append(pred_anno)
            
            imname = '%s/%s' % (image_dir, os.path.basename(true_anno.imageName))
            misc.imsave(imname, new_img)
            if i % 100 == 0:
                print(i)
    return pred_annolist, true_annolist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--test_idl', required=True)
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--logdir', default='output')
    parser.add_argument('--iou_threshold', default=0.5)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    hypes_file = '%s/hypes.json' % os.path.dirname(args.weights)
    with open(hypes_file, 'r') as f:
        H = json.load(f)
    pred_idl = '%s.%s' % (args.weights, os.path.basename(args.test_idl))
    true_idl = '%s.gt_%s' % (args.weights, os.path.basename(args.test_idl))


    pred_annolist, true_annolist = get_results(args, H)
    pred_annolist.save(pred_idl)
    true_annolist.save(true_idl)

    rpc_cmd = './utils/annolist/doRPC.py --minOverlap %f %s %s' % (args.iou_threshold, true_idl, pred_idl)
    print('$ %s' % rpc_cmd)
    rpc_output = subprocess.check_output(rpc_cmd, shell=True)
    print(rpc_output)
    txt_file = [line for line in rpc_output.split('\n') if line.strip()][-1]
    output_png = '%s/results.png' % get_image_dir(args)
    plot_cmd = './utils/annolist/plotSimple.py %s --output %s' % (txt_file, output_png)
    print('$ %s' % plot_cmd)
    plot_output = subprocess.check_output(plot_cmd, shell=True)
    print('output results at: %s' % plot_output)

if __name__ == '__main__':
    main()

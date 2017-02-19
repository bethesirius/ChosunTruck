import tensorflow as tf
import os
import json
import subprocess
import sysv_ipc
import struct

from scipy.misc import imread, imresize
from scipy import misc

from train import build_forward
from utils.annolist import AnnotationLib as al
from utils.train_utils import add_rectangles, rescale_boxes
from pymouse import PyMouse

import cv2
import argparse
import time
import numpy as np

def get_image_dir(args):
    weights_iteration = int(args.weights.split('-')[-1])
    expname = '_' + args.expname if args.expname else ''
    image_dir = '%s/images_%s_%d%s' % (os.path.dirname(args.weights), os.path.basename(args.test_boxes)[:-5], weights_iteration, expname)
    return image_dir

def get_results(args, H):
    tf.reset_default_graph()
    x_in = tf.placeholder(tf.float32, name='x_in', shape=[H['image_height'], H['image_width'], 3])
    if H['use_rezoom']:
        pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)
        grid_area = H['grid_height'] * H['grid_width']
        pred_confidences = tf.reshape(tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * H['rnn_len'], 2])), [grid_area, H['rnn_len'], 2])
        if H['reregress']:
            pred_boxes = pred_boxes + pred_boxes_deltas
    else:
        pred_boxes, pred_logits, pred_confidences = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, args.weights)

        pred_annolist = al.AnnoList()

        data_dir = os.path.dirname(args.test_boxes)
        image_dir = get_image_dir(args)
        subprocess.call('mkdir -p %s' % image_dir, shell=True)
	
	memory = sysv_ipc.SharedMemory(123463)
	memory2 = sysv_ipc.SharedMemory(123464)
	size = 768, 1024, 3

	pedal = PyMouse()
	pedal.press(1)
	road_center = 320
	while True:
	    cv2.waitKey(1)
	    frameCount = bytearray(memory.read())
	    curve = bytearray(memory2.read())
	    curve = str(struct.unpack('i',curve)[0])
	    m = np.array(frameCount, dtype=np.uint8)
	    orig_img = m.reshape(size)
	   
	    img = imresize(orig_img, (H["image_height"], H["image_width"]), interp='cubic')
	    feed = {x_in: img}
	    (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict=feed)
	    pred_anno = al.Annotation()
	    
	    new_img, rects = add_rectangles(H, [img], np_pred_confidences, np_pred_boxes,
					    use_stitching=True, rnn_len=H['rnn_len'], min_conf=args.min_conf, tau=args.tau, show_suppressed=args.show_suppressed)
	    flag = 0
	    road_center = 320 + int(curve)
	    print(road_center)
	    for rect in rects:
		print(rect.x1, rect.x2, rect.y2)
		if (rect.x1 < road_center and rect.x2 > road_center and rect.y2 > 200) and (rect.x2 - rect.x1 > 30):
			flag = 1

	    if flag is 1:
		pedal.press(2)
		print("break!")
	    else:
		pedal.release(2)
		pedal.press(1)
		print("acceleration!")
		
	    pred_anno.rects = rects
	    pred_anno.imagePath = os.path.abspath(data_dir)
	    pred_anno = rescale_boxes((H["image_height"], H["image_width"]), pred_anno, orig_img.shape[0], orig_img.shape[1])
	    pred_annolist.append(pred_anno)
	    
	    cv2.imshow('.jpg', new_img)
	    
    return none;

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='tensorbox/output/overfeat_rezoom_2017_02_09_13.28/save.ckpt-100000')
    parser.add_argument('--expname', default='')
    parser.add_argument('--test_boxes', default='default')
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--logdir', default='output')
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--tau', default=0.25, type=float)
    parser.add_argument('--min_conf', default=0.2, type=float)
    parser.add_argument('--show_suppressed', default=True, type=bool)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    hypes_file = '%s/hypes.json' % os.path.dirname(args.weights)
    with open(hypes_file, 'r') as f:
        H = json.load(f)
    expname = args.expname + '_' if args.expname else ''
    pred_boxes = '%s.%s%s' % (args.weights, expname, os.path.basename(args.test_boxes))

    get_results(args, H)

if __name__ == '__main__':
    main()

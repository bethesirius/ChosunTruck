import numpy as np
import random
import json
import os
import cv2
from scipy.misc import imread

from data_utils import (annotation_jitter, image_to_h5,
                   annotation_to_h5)
from utils.annolist import AnnotationLib as al
from rect import Rect

def rescale_boxes(anno, target_width, target_height):
    I = imread(anno.imageName)
    x_scale = target_width / float(I.shape[1])
    y_scale = target_height / float(I.shape[0])
    for r in anno.rects:
        assert r.x1 < r.x2
        r.x1 *= x_scale
        r.x2 *= x_scale
        assert r.x1 < r.x2
        r.y1 *= y_scale
        r.y2 *= y_scale
    return I

def make_sparse(n, d):
    v = np.zeros((d,), dtype=np.float32)
    v[n] = 1.
    return v

def load_idl_tf(idlfile, H, jitter):
    """Take the idlfile and net configuration and create a generator
    that outputs a jittered version of a random image from the annolist
    that is mean corrected."""

    arch = H['arch']
    annolist = al.parse(idlfile)
    annos = [x for x in annolist]
    for anno in annos:
        anno.imageName = os.path.join(
            os.path.dirname(os.path.realpath(idlfile)), anno.imageName)
    random.seed(0)
    if H['data']['truncate_data']:
        annos = annos[:10]
    while True:
        random.shuffle(annos)
        for anno in annos:
            if arch["image_width"] != 640 or arch["image_height"] != 480:
                rescale_boxes(anno, arch["image_width"], arch["image_height"])
            I = imread(anno.imageName)
            if jitter:
                jit_image, jit_anno = annotation_jitter(I,
                    anno, target_width=arch["image_width"],
                    target_height=arch["image_height"])
            else:
                I = imread(anno.imageName)
                try:
                    jit_image, jit_anno = annotation_jitter(I,
                        anno, target_width=arch["image_width"],
                        target_height=arch["image_height"],
                        jitter_scale_min=1.0, jitter_scale_max=1.0, jitter_offset=0)
                except:
                    import traceback
                    print(traceback.format_exc())
                    continue
            boxes, box_flags = annotation_to_h5(
                jit_anno, arch["grid_width"], arch["grid_height"],
                arch["rnn_len"])
            yield {"imname": anno.imageName, "raw": [], "image": jit_image,
                   "boxes": boxes, "box_flags": box_flags}

def load_data_gen(H, phase, jitter):
    arch = H["arch"]
    grid_size = arch['grid_width'] * arch['grid_height']

    data = load_idl_tf(H["data"]['%s_idl' % phase], H, jitter={'train': jitter, 'test': False}[phase])

    for d in data:
        output = {}
        
        rnn_len = arch["rnn_len"]
        box_flags = d['box_flags'][0,:,0,0:rnn_len,0]
        boxes = np.transpose(d['boxes'][0,:,:,0:rnn_len,0], (0,2,1))
        assert(box_flags.shape == (grid_size, rnn_len))
        assert(boxes.shape == (grid_size, rnn_len, 4))

        output['image'] = d['image']
        output['confs'] = np.array([make_sparse(row[0], d=2) for row in box_flags])
        output['boxes'] = boxes
        output['flags'] = box_flags
        
        yield output

def add_rectangles(orig_image, confidences, boxes, arch, use_stitching=False, rnn_len=1, min_conf=0.5):
    image = np.copy(orig_image[0])
    num_cells = arch["grid_height"] * arch["grid_width"]
    boxes_r = np.reshape(boxes, (arch["batch_size"],
                                 arch["grid_height"],
                                 arch["grid_width"],
                                 rnn_len,
                                 4))
    confidences_r = np.reshape(confidences, (arch["batch_size"],
                                             arch["grid_height"],
                                             arch["grid_width"],
                                             rnn_len,
                                             2))
    cell_pix_size = 32
    all_rects = [[[] for _ in range(arch["grid_width"])] for _ in range(arch["grid_height"])]
    for n in range(rnn_len):
        for y in range(arch["grid_height"]):
            for x in range(arch["grid_width"]):
                bbox = boxes_r[0, y, x, n, :]
                conf = confidences_r[0, y, x, n, 1]
                abs_cx = int(bbox[0]) + cell_pix_size/2 + cell_pix_size * x
                abs_cy = int(bbox[1]) + cell_pix_size/2 + cell_pix_size * y
                w = bbox[2]
                h = bbox[3]
                all_rects[y][x].append(Rect(abs_cx,abs_cy,w,h,conf))

    if use_stitching:
        from stitch_wrapper import stitch_rects
        acc_rects = stitch_rects(all_rects)
    else:
        acc_rects = [r for row in all_rects for cell in row for r in cell]


    for rect in acc_rects:
        if rect.confidence > min_conf:
            cv2.rectangle(image,
                (rect.cx-int(rect.width/2), rect.cy-int(rect.height/2)),
                (rect.cx+int(rect.width/2), rect.cy+int(rect.height/2)),
                (0,255,0),
                2)

    return image

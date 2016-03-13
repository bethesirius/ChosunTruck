import numpy as np
import random
import json
import os
import cv2
import itertools
from scipy.misc import imread, imresize

from data_utils import (annotation_jitter, image_to_h5,
                   annotation_to_h5)
from utils.annolist import AnnotationLib as al
from rect import Rect

def rescale_boxes(current_shape, anno, target_height, target_width):
    x_scale = target_width / float(current_shape[1])
    y_scale = target_height / float(current_shape[0])
    for r in anno.rects:
        assert r.x1 < r.x2
        r.x1 *= x_scale
        r.x2 *= x_scale
        assert r.x1 < r.x2
        r.y1 *= y_scale
        r.y2 *= y_scale
    return anno

def load_idl_tf(idlfile, H, jitter):
    """Take the idlfile and net configuration and create a generator
    that outputs a jittered version of a random image from the annolist
    that is mean corrected."""

    annolist = al.parse(idlfile)
    annos = []
    for anno in annolist:
        anno.imageName = os.path.join(
            os.path.dirname(os.path.realpath(idlfile)), anno.imageName)
        annos.append(anno)
    random.seed(0)
    if H['data']['truncate_data']:
        annos = annos[:10]
    for epoch in itertools.count():
        random.shuffle(annos)
        for anno in annos:
            I = imread(anno.imageName)
            if I.shape[2] == 4:
                I = I[:, :, :3]
            if I.shape[0] != H["arch"]["image_height"] or I.shape[1] != H["arch"]["image_width"]:
                if epoch == 0:
                    anno = rescale_boxes(I.shape, anno, H["arch"]["image_height"], H["arch"]["image_width"])
                I = imresize(I, (H["arch"]["image_height"], H["arch"]["image_width"]), interp='cubic')
            if jitter:
                jitter_scale_min=0.9
                jitter_scale_max=1.1
                jitter_offset=16
                I, anno = annotation_jitter(I,
                                            anno, target_width=H["arch"]["image_width"],
                                            target_height=H["arch"]["image_height"],
                                            jitter_scale_min=jitter_scale_min,
                                            jitter_scale_max=jitter_scale_max,
                                            jitter_offset=jitter_offset)

            boxes, flags = annotation_to_h5(anno,
                                            H["arch"]["grid_width"],
                                            H["arch"]["grid_height"],
                                            H["arch"]["rnn_len"])

            yield {"image": I, "boxes": boxes, "flags": flags}

def make_sparse(n, d):
    v = np.zeros((d,), dtype=np.float32)
    v[n] = 1.
    return v

def load_data_gen(H, phase, jitter):
    grid_size = H["arch"]['grid_width'] * H["arch"]['grid_height']

    data = load_idl_tf(H["data"]['%s_idl' % phase], H, jitter={'train': jitter, 'test': False}[phase])

    for d in data:
        output = {}
        
        rnn_len = H["arch"]["rnn_len"]
        flags = d['flags'][0,:,0,0:rnn_len,0]
        boxes = np.transpose(d['boxes'][0,:,:,0:rnn_len,0], (0,2,1))
        assert(flags.shape == (grid_size, rnn_len))
        assert(boxes.shape == (grid_size, rnn_len, 4))

        output['image'] = d['image']
        output['confs'] = np.array([[make_sparse(detection, d=2) for detection in cell] for cell in flags])
        output['boxes'] = boxes
        output['flags'] = flags
        
        yield output

def add_rectangles(orig_image, confidences, boxes, arch, use_stitching=False, rnn_len=1, min_conf=0.5):
    image = np.copy(orig_image[0])
    num_cells = arch["grid_height"] * arch["grid_width"]
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

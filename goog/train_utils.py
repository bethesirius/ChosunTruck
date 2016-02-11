import numpy as np
import random
import json
import os
from scipy.misc import imread, imresize
from utils import (annotation_jitter, image_to_h5,
                   annotation_to_h5, load_data_mean)
from utils.annolist import AnnotationLib as al

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
    return []

def load_idl(idlfile, data_mean, net_config, jitter):
    """Take the idlfile, data mean and net configuration and create a generator
    that outputs a jittered version of a random image from the annolist
    that is mean corrected."""

    annolist = al.parse(idlfile)
    annos = [x for x in annolist]
    for anno in annos:
        anno.imageName = os.path.join(
            os.path.dirname(os.path.realpath(idlfile)), anno.imageName)
    random.seed(0)
    random.shuffle(annos)
    result = []
    for anno in annos:
        #I = rescale_boxes(anno, net_config["img_width"], net_config["img_height"])
        #if jitter:
            #jit_image, jit_anno = annotation_jitter(I,
                #anno, target_width=net_config["img_width"],
                #target_height=net_config["img_height"])
        #else:
            #jit_image = I
            #jit_anno = anno
        #image = image_to_h5(jit_image, data_mean, image_scaling=1.0)
        boxes, box_flags = annotation_to_h5(
            anno, net_config["grid_width"], net_config["grid_height"],
            net_config["region_size"], net_config["max_len"])
        #if net_config.get("use_log", False):
            #boxes[:, :, 2:4, :, :] = np.log(boxes[:, :, 2:4, :, :])
        yield {"imname": anno.imageName, "raw": [], "image": [],
               "boxes": boxes, "box_flags": box_flags}
        #yield {"imname": anno.imageName, "raw": jit_image, "image": image,
               #"boxes": boxes, "box_flags": box_flags}

def make_sparse(n, d):
    v = np.zeros((d,), dtype=np.float32)
    v[n] = 1.
    return v

def load_data(load_fast):
    config = json.load(open('./reinspect/config.json', 'r'))
    net_config = config["net"]
    data_mean = np.ones((480, 640, 3)) * 128
    if load_fast:
        from itertools import islice
        a = list(islice(load_idl('./data/brainwash/brainwash_train.idl', data_mean, net_config, False), 20))
    else:
        a = list(load_idl('./data/brainwash/brainwash_train.idl', data_mean, net_config, False))

    output = {}
    for phase in ['train', 'test']:
        #with open('%s/brainwash_%s.idl' % (base_dir, phase)) as f:
            #data = [line.strip() for line in f.readlines()]
        images = []
        labels = []
        box_labels = []
        for idx, d in enumerate(a):
            images.append(d['imname'].replace('jpg', 'png'))
            #'%s/%s' % (base_dir, line.split('"')[1].replace('jpg', 'png')))
            #labels.append(min(sum(1 for char in line if char == ')'), 9))
            flags = d['box_flags'][0,:,0,0:1,0]
            boxes = np.transpose(d['boxes'][0,:,:,0:1,0], (0,2,1))
            assert(flags.shape == (300, 1))
            assert(boxes.shape == (300, 1, 4))
            labels.append([make_sparse(row[0], d=10) for row in flags]) #TODO: change d to 2 and retrain
            box_labels.append(boxes)
            #import ipdb; ipdb.set_trace()
            
        labels_array = np.array(labels)
        box_labels_array = np.array(box_labels)
        output[phase] = {'Y': labels_array, 'boxes': box_labels_array, 'X': images}
    return output


#load_data('')


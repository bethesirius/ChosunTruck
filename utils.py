import numpy as np
import random

def load_data(directory):
    base_dir = '/home/stewartr/git/reinspect/data/brainwash'
    output = {}
    for phase in ['train', 'test']:
        with open('%s/brainwash_%s.idl' % (base_dir, phase)) as f:
            data = [line.strip() for line in f.readlines()]
        images = []
        labels = []
        for idx, line in enumerate(data):
            images.append('%s/%s' % (base_dir, line.split('"')[1].replace('jpg', 'png')))
            labels.append(min(sum(1 for char in line if char == ')'), 9))
            
        output[phase] = {'Y': labels, 'X': images}
    return output

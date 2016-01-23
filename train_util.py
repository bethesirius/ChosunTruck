import numpy as np
import random

filenames = [
    '/home/stewartr/git/reinspect/data/brainwash/brainwash_10_27_2014_images/00002000_640x480.png',
    '/home/stewartr/git/reinspect/data/brainwash/brainwash_10_27_2014_images/00004000_640x480.png',
    '/home/stewartr/git/reinspect/data/brainwash/brainwash_10_27_2014_images/00005000_640x480.png',
    '/home/stewartr/git/reinspect/data/brainwash/brainwash_10_27_2014_images/00006000_640x480.png',
    '/home/stewartr/git/reinspect/data/brainwash/brainwash_10_27_2014_images/00007000_640x480.png',
    '/home/stewartr/git/reinspect/data/brainwash/brainwash_10_27_2014_images/00008000_640x480.png',
    '/home/stewartr/git/reinspect/data/brainwash/brainwash_10_27_2014_images/00009000_640x480.png',
    '/home/stewartr/git/reinspect/data/brainwash/brainwash_10_27_2014_images/00010000_640x480.png',
    '/home/stewartr/git/reinspect/data/brainwash/brainwash_10_27_2014_images/00011000_640x480.png',
]

def load_data(directory):
    data_length = len(filenames * 10)
    images = filenames * 10
    targets = [random.randrange(2) for _ in range(data_length)]
    train = {'Y': targets, 'X': images}
    test = train
    return {'train': train, 'test': test}

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
            #print(labels[-1])
            
        output[phase] = {'Y': labels, 'X': images}
    return output
    #train = {'Y': labels, 'X': images}
    #test = train
    #return {'train': train, 'test': test}

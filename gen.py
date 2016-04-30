import sys
import random
import argparse
import subprocess
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('--gpu')
parser.add_argument('--hypes')
parser.add_argument('--expname', default='default')
args = parser.parse_args()
basepath = os.path.basename(args.hypes)
subprocess.call('mkdir -p hypes_gen/%s' % args.expname, shell=True)
for i in range(10):
    H = json.load(open(args.hypes, 'r'))
    H['solver']['learning_rate'] *= 10 ** (random.random() - 0.5)
    H['solver']['head_weights'][0] *= 3 ** (random.random() - 0.5)
    H['solver']['max_iter'] = 20000
    json_file = 'hypes_gen/%s.%d.json' % (basepath[:-5], i)
    with open(json_file, 'w') as f:
        f.write(json.dumps(H))
    train_cmd = 'python train.py --gpu %s --hypes %s --logdir output_kitti_gen' % (args.gpu, json_file)
    subprocess.check_output(train_cmd, shell=True)

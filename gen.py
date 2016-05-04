import sys
import socket
import random
import argparse
import subprocess
import os
import json

import threading

class run_exp(threading.Thread):
    def __init__ (self, host, gpu, expname, job_num, hypes):
        threading.Thread.__init__(self)
        self.host = host
        self.gpu = gpu
        self.expname = expname
        self.job_num = job_num
        self.hypes = hypes

    def run(self):
        host = self.host
        gpu = self.gpu
        expname = self.expname
        job_num = self.job_num
        hypes = self.hypes
        basepath = os.path.basename(hypes)

        for i in range(10):
            H = json.load(open(hypes, 'r'))
            H['solver']['learning_rate'] *= 10 ** (random.random() - 0.5)
            H['solver']['head_weights'][0] *= 3 ** (random.random() - 0.5)
            H['solver']['max_iter'] = 100001
            json_file = 'hypes_gen/%s/%s.%s.%d.%d.%s.json' % (expname, basepath[:-5], expname, i, int(gpu), host)
            with open(json_file, 'w') as f:
                f.write(json.dumps(H))
            train_path = os.path.dirname(os.path.realpath(__file__))
            train_cmd = 'ssh %s "cd %s && python train.py --gpu %s --hypes %s --logdir output_kitti_gen/%s"' % (host, train_path, gpu, json_file, expname)
            subprocess.check_output(train_cmd, shell=True)
            last_ckpt = H['solver']['max_iter'] - H['solver']['max_iter'] % H['logging']['save_iter']
            eval_cmd = 'ssh %s "cd %s && python evaluate.py --gpu %d --test_idl data/kitti/val_small_unscale.idl --iou_threshold 0.5 --weights output_kitti_gen/%s/%s_*/save.ckpt-%d"' % (host, train_path, gpu, expname, os.path.basename(json_file)[:-5], last_ckpt)
            subprocess.check_output(eval_cmd, shell=True)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0)
parser.add_argument('--hypes', required=True)
parser.add_argument('--expname', required=True)
parser.add_argument('--hosts', default=socket.gethostname())


args = parser.parse_args()
subprocess.call('mkdir -p hypes_gen/%s' % args.expname, shell=True)
job_num = 0
threads = []
for host in args.hosts.split(','):
    for gpu_id in map(int, args.gpu.split(',')):
        thread = run_exp(host, gpu_id, args.expname, job_num, args.hypes)
        thread.start()
        threads.append(thread)
        job_num += 1

for thread in threads:
    thread.join()
        #thread.start_new_thread(run, (host, gpu_id, args.expname, job_num, args.hypes))
        #run(host, gpu_id, args.expname, job_num, args.hypes)

<img src=http://russellsstewart.com/s/ReInspect_output.jpg></img>

Tensordetect is a simple framework for training neural networks to detect objects in images. 
Training requires a text file (see [here](http://russellsstewart.com/s/tensordetect/brainwash_test.txt), for example)
of paths to images on disk and the corresponding object locations in each image.
The basic model implements the simple and robust GoogLeNet-OverFeat algorithm. We additionally provide an implementation of the 
[ReInspect](https://github.com/Russell91/ReInspect/)
algorithm, achieving state-of-the-art detection results on the TUD crossing and brainwash datasets. 

Special thanks to [Brett Kuprel](http://stanford.edu/~kuprel/) of Sebastian Thrun's group for providing the initial code
to hack into and finetune Google's pretrained ImageNet weights.

## Overfeat Installation & Training
First, [install tensorflow from source or pip](https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html#pip-installation)
    
    $ git clone http://github.com/russell91/tensordetect
    $ cd tensordetect
    $ ./download_data.sh
    $ python train.py --hypes hypes/default.json --gpu 0 --logdir output

Note that running on your own dataset should only require modifyngi the `hypes/default.json` file. 
When finished training, you can use code from the provided 
[ipython notebook](https://github.com/Russell91/tensordetect/blob/master/evaluate.ipynb)
to get results on your test set.

## ReInspect Installation & training

ReInspect, [initially implemented](https://github.com/Russell91/ReInspect/edit/master/README.md) in Caffe,
is an neural network extension to Overfeat-GoogLeNet in Tensorflow.
It is designed for high performance object detection in images with heavily overlapping instances.
See <a href="http://arxiv.org/abs/1506.04878" target="_blank">the paper</a> for details or the <a href="https://www.youtube.com/watch?v=QeWl0h3kQ24" target="_blank">video</a> for a demonstration.

    $ git clone http://github.com/russell91/tensordetect
    $ cd tensordetect
    $ ./download_data.sh
    
    # Install tensorflow from source
    $ git clone --recurse-submodules https://github.com/tensorflow/tensorflow
    # Add code for the custom hungarian layer user_op
    $ cp /path/to/tensordetect/data/lstm/hungarian.cc /path/to/tensorflow/tensorflow/core/user_ops/
    # Proceed with the GPU installation of tensorflow from source...
    # (see https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html#installing-from-sources)
    $ cd /path/to/tensordetect/utils && make && cd ..
    $ python train.py --hypes hypes/lstm.json --gpu 0 --logdir output

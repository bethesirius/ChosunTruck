<img src=http://russellsstewart.com/s/tensorbox/tensorbox_output.jpg></img>

TensorBox is a simple framework for training neural networks to detect objects in images. 
Training requires a text file (see [here](http://russellsstewart.com/s/tensorbox/brainwash_test.txt), for example)
of paths to images on disk and the corresponding object locations in each image.
The basic model implements the simple and robust GoogLeNet-OverFeat algorithm. We additionally provide an implementation of the 
[ReInspect](https://github.com/Russell91/ReInspect/)
algorithm, reproducing state-of-the-art detection results on the highly occluded TUD crossing and brainwash datasets. 

Special thanks to [Brett Kuprel](http://stanford.edu/~kuprel/) of Sebastian Thrun's group for providing the initial code
to hack into Google's pretrained ImageNet weights for finetuning.

## OverFeat Installation & Training
First, [install TensorFlow from source or pip](https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html#pip-installation)
    
    $ git clone http://github.com/russell91/tensorbox
    $ cd tensorbox
    $ ./download_data.sh
    $ python train.py --hypes hypes/overfeat_rezoom.json --gpu 0 --logdir output

Note that running on your own dataset should only require modifying the `hypes/overfeat_rezoom.json` file. 
When finished training, you can use code from the provided 
[ipython notebook](https://github.com/Russell91/tensorbox/blob/master/evaluate.ipynb)
to get results on your test set.

## ReInspect Installation & Training

ReInspect, [initially implemented](https://github.com/Russell91/ReInspect) in Caffe,
is a neural network extension to Overfeat-GoogLeNet in Tensorflow.
It is designed for high performance object detection in images with heavily overlapping instances.
See <a href="http://arxiv.org/abs/1506.04878" target="_blank">the paper</a> for details or the <a href="https://www.youtube.com/watch?v=QeWl0h3kQ24" target="_blank">video</a> for a demonstration.

    $ git clone http://github.com/russell91/tensorbox
    $ cd tensorbox
    $ ./download_data.sh
    
    $ # Install tensorflow from source
    $ git clone --recurse-submodules https://github.com/tensorflow/tensorflow
    $ # Add code for the custom hungarian layer user_op
    $ cp /path/to/tensorbox/utils/hungarian/hungarian.cc /path/to/tensorflow/tensorflow/core/user_ops/
    $ # Proceed with the GPU installation of tensorflow from source...
    $ # (see https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html#installing-from-sources)

    $ cd /path/to/tensorbox/utils && make && cd ..
    $ python train.py --hypes hypes/lstm_rezoom.json --gpu 0 --logdir output

## Tensorboard

You can visualize the progress of your experiments during training using Tensorboard.

    $ cd /path/to/tensorbox
    $ tensorboard --logdir output
    $ # (optional, start an ssh tunnel if not experimenting locally)
    $ ssh myserver -N -L localhost:6006:localhost:6006
    $ # open localhost:6006 in your browser
    
For example, the following is a screenshot of a Tensorboard comparing two different experiments with learning rate decays that kick in at different points. The learning rate drops in half at 60k iterations for the green experiment and 300k iterations for red experiment.
    
<img src=http://russellsstewart.com/s/tensorbox/tensorboard_loss.png></img>

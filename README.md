<img src=http://russellsstewart.com/s/tensorbox/tensorbox_output.jpg></img>

TensorBox is a simple framework for training neural networks to detect objects in images. 
Training requires a json file (e.g. [here](http://russellsstewart.com/s/tensorbox/test_boxes.json))
containing a list of images and the bounding boxes in each image.
The basic model implements the simple and robust GoogLeNet-OverFeat algorithm. We additionally provide an implementation of the 
[ReInspect](https://github.com/Russell91/ReInspect/)
algorithm, reproducing state-of-the-art detection results on the highly occluded TUD crossing and brainwash datasets.

## OverFeat Installation & Training
First, [install TensorFlow from source or pip](https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html#pip-installation)
    
    $ git clone http://github.com/russell91/tensorbox
    $ cd tensorbox
    $ ./download_data.sh
    $ cd /path/to/tensorbox/utils && make && cd ..
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

    # REQUIRES TENSORFLOW VERSION >= 0.8
    $ git clone http://github.com/russell91/tensorbox
    $ cd tensorbox
    $ ./download_data.sh
    
    $ # Download the cudnn version used by your tensorflow verion and 
    $ # put the libcudnn*.so files on your LD_LIBRARY_PATH e.g.
    $ cp /path/to/appropriate/cudnn/lib64/* /usr/local/cuda/lib64

    $ cd /path/to/tensorbox/utils && make && make hungarian && cd ..
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

<img src=http://russellsstewart.com/s/tensorbox/tensorbox_output.jpg></img>

### Tensorflow == 0.11 supported by current version (Please report if you use >0.11 and get errors).

TensorBox is a simple framework for training neural networks to detect objects in images. 
Training requires a json file (e.g. [here](http://russellsstewart.com/s/tensorbox/test_boxes.json))
containing a list of images and the bounding boxes in each image.
The basic model implements the simple and robust GoogLeNet-OverFeat algorithm. We additionally provide an implementation of the 
[ReInspect](https://github.com/Russell91/ReInspect/)
algorithm, reproducing state-of-the-art detection results on the highly occluded TUD crossing and brainwash datasets.

## OverFeat Installation & Training
First, [install TensorFlow from source or pip](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#pip-installation) (NB: source installs currently break threading on 0.11)
    
    $ git clone http://github.com/russell91/tensorbox
    $ cd tensorbox
    $ ./download_data.sh
    $ cd /path/to/tensorbox/utils && make && cd ..
    $ python train.py --hypes hypes/overfeat_rezoom.json --gpu 0 --logdir output
    $ #see evaluation instructions below

Note that running on your own dataset should only require modifying the `hypes/overfeat_rezoom.json` file. 

## ReInspect Installation & Training

ReInspect, [initially implemented](https://github.com/Russell91/ReInspect) in Caffe,
is a neural network extension to Overfeat-GoogLeNet in Tensorflow.
It is designed for high performance object detection in images with heavily overlapping instances.
See <a href="http://arxiv.org/abs/1506.04878" target="_blank">the paper</a> for details or the <a href="https://www.youtube.com/watch?v=QeWl0h3kQ24" target="_blank">video</a> for a demonstration.

    # REQUIRES TENSORFLOW VERSION >= 0.11
    $ git clone http://github.com/russell91/tensorbox
    $ cd tensorbox
    $ ./download_data.sh
    
    $ # Download the cudnn version used by your tensorflow verion and 
    $ # put the libcudnn*.so files on your LD_LIBRARY_PATH e.g.
    $ cp /path/to/appropriate/cudnn/lib64/* /usr/local/cuda/lib64

    $ cd /path/to/tensorbox/utils && make && make hungarian && cd ..
    $ python train.py --hypes hypes/lstm_rezoom.json --gpu 0 --logdir output
    $ #see evaluation instructions below

## Evaluation

There are two options for evaluation, an ipython notebook and a python script.

### IPython Notebook
The [ipython notebook](https://github.com/Russell91/tensorbox/blob/master/evaluate.ipynb)
allows you to interactively modify the inference algorithm, and can be run concurrently
with training (assuming you have 2 gpus). You can evaluate on new data by modifying paths
and pointing to new weights.

### Python script
For those who would prefer to evaluate using a script, you can alternately use evaluate.py.
The following instructions demonstrate how evaluate.py wase used after one of my experiments -
you will need to change paths as appropriate:

    $ # kill training script if you don't have a spare GPU
    $ cd /path/to/tensorbox
    $ python evaluate.py --weights output/overfeat_rezoom_2017_01_17_15.20/save.ckpt-130000 --test_boxes data/brainwash/val_boxes.json
    $ # val_boxes should contain the list of images you want to output boxes on, and
    $ # the annotated boxes for each image if you want to generate a precision recall curve
    $ cd ./output/overfeat_rezoom_2017_01_17_15.20/images_val_boxes_130000/
    $ ls # ... notice the images with predicted boxes painted on, and the results saved in results.png
    $ python -m SimpleHTTPServer 8080 # set up a image server to view the images from your browser
    $ ssh myserver -N -L localhost:8080:localhost:8080 # set up an ssh tunnel to your server (skip if running locally)
    $ # open firefox and visit localhost:8080 to view images

## Finetuning

If you get some decent results and want to improve your performance, there are many things you can try.
For hyperparameter optimization, the Learning rate, dropout ratios, and parameter initializations are a great place to start. You may want to
read this <a href="http://russellsstewart.com/blog/0">blog post</a> for a more generic tutorial on debugging neural nets.
We have recently added a resnet version as well, which should work slightly better on larger boxes (this repo has historically done poorly
on these, as they weren't port of the original research goal). I would recommend using the overfeat version over the lstm as well
if you have a large variation in box sizes.

## Tensorboard

You can visualize the progress of your experiments during training using Tensorboard.

    $ cd /path/to/tensorbox
    $ tensorboard --logdir output
    $ # (optional, start an ssh tunnel if not experimenting locally)
    $ ssh myserver -N -L localhost:6006:localhost:6006
    $ # open localhost:6006 in your browser
    
For example, the following is a screenshot of a Tensorboard comparing two different experiments with learning rate decays that kick in at different points. The learning rate drops in half at 60k iterations for the green experiment and 300k iterations for red experiment.
    
<img src=http://russellsstewart.com/s/tensorbox/tensorboard_loss.png></img>

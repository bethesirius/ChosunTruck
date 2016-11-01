import inception_v1 as inception
import tensorflow.contrib.slim as slim

def model(x, H, reuse, is_training=True):
    with slim.arg_scope(inception.inception_v1_arg_scope()):
        _, T = inception.inception_v1(x,
                                      is_training=is_training,
                                      num_classes=1001,
                                      spatial_squeeze=False,
                                      reuse=reuse)
    coarse_feat = T['Mixed_5b']

    # fine feat can be used to reinspect input
    attention_lname = H.get('attention_lname', 'Mixed_3b')
    early_feat = T[attention_lname]
    early_feat_channels = 480

    return coarse_feat, early_feat, early_feat_channels

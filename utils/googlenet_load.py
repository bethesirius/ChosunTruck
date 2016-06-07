import tensorflow as tf
from kaffe import mynet
import os
import numpy as np

def init(H, config=None):
    if config is None:
        gpu_options = tf.GPUOptions()
        config = tf.ConfigProto(gpu_options=gpu_options)

    k = H['arch']['num_classes']
    features_dim = 1024
    input_layer = 'input'

    features_layers = ['output/confidences', 'output/boxes']

    graph_def_orig_file = '%s/../data/googlenet.pb' % os.path.dirname(os.path.realpath(__file__))

    dense_layer_num_output = [k, 4]

    googlenet_graph = tf.Graph()
    graph_def = tf.GraphDef()
    tf.set_random_seed(0)
    with open(graph_def_orig_file) as f:
        tf.set_random_seed(0)
        graph_def.MergeFromString(f.read())

    with googlenet_graph.as_default():
        tf.import_graph_def(graph_def, name='')

    input_op = googlenet_graph.get_operation_by_name(input_layer)

    weights_ops = [
        op for op in googlenet_graph.get_operations() 
        if any(op.name.endswith(x) for x in [ '_w', '_b'])
        and op.type == 'Const'
    ]

    reuse_ops = [
        op for op in googlenet_graph.get_operations() 
        if op not in weights_ops + [input_op]
        and op.name != 'output'
    ]

    with tf.Session(graph=googlenet_graph, config=config):
        weights_orig = {
            op.name: op.outputs[0].eval()
            for op in weights_ops
        }

    def weight_init(num_output):
        return 0.001 * np.random.randn(features_dim, num_output).astype(np.float32)

    def bias_init(num_output):
        return 0.001 * np.random.randn(num_output).astype(np.float32)


    W = [
        tf.Variable(weight_init(dense_layer_num_output[i]), 
                    name='softmax/weights_{}'.format(i)) 
        for i in range(len(features_layers))
    ]

    B = [
        tf.Variable(bias_init(dense_layer_num_output[i]),
                    name='softmax/biases_{}'.format(i)) 
        for i in range(len(features_layers))
    ]

    weight_vars = {
        name: tf.Variable(weight, name=name)
        for name, weight in weights_orig.iteritems()
    }

    weight_tensors = {
        name: tf.convert_to_tensor(weight)
        for name, weight in weight_vars.iteritems()
    }

    W_norm = [tf.nn.l2_loss(weight) for weight in weight_vars.values() + W]
    W_norm = tf.reduce_sum(tf.pack(W_norm), name='weights_norm')
    tf.scalar_summary(W_norm.op.name, W_norm)

    googlenet = {
        "W": W,
        "B": B,
        "weight_tensors": weight_tensors,
        "reuse_ops": reuse_ops,
        "input_op": input_op,
        "W_norm": W_norm,
        }
    return googlenet

def model(x, googlenet, H):
    weight_tensors = googlenet["weight_tensors"]
    input_op = googlenet["input_op"]
    reuse_ops = googlenet["reuse_ops"]
    def is_early_loss(name):
        early_loss_layers = ['head0', 'nn0', 'softmax0', 'head1', 'nn1', 'softmax1', 'output1']
        return any(name.startswith(prefix) for prefix in early_loss_layers)

    T = weight_tensors
    T[input_op.name] = x

    for op in reuse_ops:
        if is_early_loss(op.name):
            continue
        elif op.name == 'avgpool0':
            pool_op = tf.nn.avg_pool(T['mixed5b'], ksize=[1,H['arch']['grid_height'],H['arch']['grid_width'],1], strides=[1,1,1,1], padding='VALID', name=op.name)
            T[op.name] = pool_op

        else:
            copied_op = x.graph.create_op(
                op_type = op.type, 
                inputs = [T[t.op.name] for t in list(op.inputs)], 
                dtypes = [o.dtype for o in op.outputs], 
                name = op.name, 
                attrs =  op.node_def.attr
            )

            T[op.name] = copied_op.outputs[0]
            #T[op.name] = tf.Print(copied_op.outputs[0], [tf.shape(copied_op.outputs[0]), tf.constant(op.name)], summarize=4)
    

    coarse_feat = T['mixed5b']

    # fine feat can be used to reinspect input
    attention_lname = H['arch'].get('attention_lname', 'mixed3b')
    early_feat = T[attention_lname]
    early_feat_channels = 480

    return coarse_feat, early_feat, early_feat_channels

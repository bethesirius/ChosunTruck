import tensorflow as tf
import numpy as np
from numpy import random as rnd
import shutil
from train_util import load_data
import time

learning_rate = 0.01
weight_init_size = 1e-3
batch_size = {
    'train': 50,
    'test': 50
}
num_epochs = 100
weight_decay = 1e-5
num_threads = 6
summary_dir = 'inceptionv3'
restore_file = None
save_file = 'inceptionv3/save.ckpt'
graph_def_orig_file = 'inception_params/tensorflow_inception_graph.pb'
image_size = 299
input_mean = 128.
input_std = 128.

data = load_data('../data/')
k = len(set(data['train']['Y']))

graphs = {name: tf.Graph() for name in ['orig', 'deploy']}

graph_def = tf.GraphDef()
with open(graph_def_orig_file) as f:
    graph_def.MergeFromString(f.read())

with graphs['orig'].as_default():
    tf.import_graph_def(graph_def, name='')

input_op = graphs['orig'].get_operation_by_name('Mul')
logits_op = graphs['orig'].get_operation_by_name('pool_3')

param_ops = [
    op for op in graphs['orig'].get_operations() 
    if ('params' in op.name or 'batchnorm/beta' in op.name)
    and op.type == 'Const'
]

reuse_ops = graphs['orig'].get_operations()
i_input = reuse_ops.index(input_op)
i_logits = reuse_ops.index(logits_op)
reuse_ops = reuse_ops[i_input+1:i_logits+1]
reuse_ops = [op for op in reuse_ops if op not in param_ops]

with tf.Session(graph=graphs['orig']):
    
    weights_orig = {
        op.name: op.outputs[0].eval()
        for op in param_ops
    }

weight_init = lambda: weight_init_size*rnd.randn(2048, k).astype(np.float32)
w = tf.Variable(weight_init(), name='softmax/weights')

bias_init = lambda: weight_init_size*rnd.randn(k).astype(np.float32)
b = tf.Variable(bias_init(), name='softmax/biases')

weight_vars = {
    name: tf.Variable(weight, name=name)
    for name, weight in weights_orig.iteritems()
}

def forward(X, Y, batch_size, phase):

    x, y = tf.train.slice_input_producer(
        tensor_list = [X, Y],
        shuffle = True,
        capacity = batch_size*(num_threads+1)
    )

    x = tf.read_file(x)
    x = tf.image.decode_png(x, channels=3)
    if phase == 'train':
        x = tf.image.resize_images(x, image_size/5*6, image_size/5*6)
        tf.image_summary('orig_image', tf.expand_dims(x, 0))
        x = tf.image.random_crop(x, [image_size, image_size])
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)
#         x = tf.image.random_contrast(x, lower=0.8, upper=1.2)
        x0 = x
        tf.image_summary('train_image', tf.expand_dims(x, 0))
    else:
        x = tf.image.resize_images(x, image_size, image_size)

    x -= input_mean
    x /= input_std

    y = tf.sparse_to_dense(y, [k], 1.0, 0)

    x, y = tf.train.shuffle_batch(
        tensor_list = [x, y],
        batch_size = batch_size,
        num_threads = num_threads,
        capacity = batch_size*(num_threads+1),
        min_after_dequeue = batch_size
    )

    tensors = {
        name: tf.convert_to_tensor(weight)
        for name, weight in weight_vars.iteritems()
    }
    tensors[input_op.name] = x

    for op in reuse_ops:

        if op.type == 'BatchNormWithGlobalNormalization':

            t, mu, sig, beta, gamma = [tensors[t.op.name] for t in list(op.inputs)]
            mu, sig = tf.nn.moments(t, [0,1,2])
            tensors[op.name] = tf.nn.batch_norm_with_global_normalization(
                t=t, m=mu, v=sig, beta=beta, gamma=gamma, 
                variance_epsilon=1e-8, scale_after_normalization=True, 
                name=phase+'/'+op.name
            )

        else:

            copied_op = tf.get_default_graph().create_op (
                op_type = op.type, 
                inputs = [tensors[t.op.name] for t in list(op.inputs)], 
                dtypes = [o.dtype for o in op.outputs], 
                name = phase + '/' + op.name, 
                attrs =  op.node_def.attr
            )

            tensors[op.name] = copied_op.outputs[0]

    z = tf.squeeze(tensors[logits_op.name])
    z = tf.nn.xw_plus_b(z,w,b, name=phase+'/logits')
    L = tf.nn.softmax_cross_entropy_with_logits(z, y)
    L = tf.reduce_mean(L, name=phase+'/loss')

    a = tf.equal(tf.argmax(y, 1), tf.argmax(z, 1))
    a = tf.reduce_mean(tf.cast(a, 'float'), name=phase+'/accuracy')

    B = tf.shape(tensors[logits_op.name])
    return L, a, B

L, a, B = {}, {}, {}

global_step = tf.Variable(0, trainable=False)

for phase in ['train', 'test']:
    L[phase], a[phase], B[phase] = \
        forward(data[phase]['X'], data[phase]['Y'], batch_size[phase], phase)
    tf.scalar_summary(L[phase].op.name, L[phase])
    tf.scalar_summary(a[phase].op.name, a[phase])

moving_avg = tf.train.ExponentialMovingAverage(0.99)
smooth_op = moving_avg.apply([a['test'], L['test']])
tf.scalar_summary(a['test'].op.name + '/smooth', moving_avg.average(a['test']))
tf.scalar_summary(L['test'].op.name + '/smooth', moving_avg.average(L['test']))

weights = weight_vars.values() + [w]
W_norm = [tf.nn.l2_loss(weight) for weight in weights]
W_norm = tf.reduce_sum(tf.pack(W_norm), name='weights_norm')
tf.scalar_summary(W_norm.op.name, W_norm)

# opt = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=0.1)
opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate, epsilon=1.0, decay=0.9)

grads = opt.compute_gradients(L['train'] + weight_decay*W_norm)
train_op = opt.apply_gradients(grads, global_step=global_step)
summary_op = tf.merge_all_summaries()

coord = tf.train.Coordinator()

writer = tf.train.SummaryWriter(
    logdir=summary_dir, 
    flush_secs=10
    )
writer.add_graph(tf.get_default_graph().as_graph_def())

saver = tf.train.Saver()

config = tf.ConfigProto(
    log_device_placement=True,
    device_count={'CPU': 6, 'GPU': 3},
)

with tf.Session(config=config) as sess:

    tf.initialize_all_variables().run()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    if restore_file: saver.restore(sess, restore_file)

    pstr = 'Exp Train Loss: %.2f, Test Accuracy: %.1f%%, W_norm/Loss: %.2f, Time/image (ms): %.1f'

    try:

        while not coord.should_stop():

            t = time.time()
            train_loss, test_accuracy, weights_norm, summary_str, _, _, shape = \
                sess.run([L['train'], a['test'], W_norm, summary_op, train_op, smooth_op, B['train']])
            print(shape)
            dt = (time.time()-t)/batch_size['train']

            writer.add_summary(summary_str, global_step=global_step.eval())
            print pstr%(np.exp(train_loss), test_accuracy*100, weight_decay*weights_norm, dt*1000)

            if global_step.eval()%10 == 0: saver.save(sess, save_file, global_step=global_step)

    except tf.errors.OutOfRangeError:
        print('Epoch limit reached')
    finally:
        print('Done training')
        coord.request_stop()

coord.join(threads)

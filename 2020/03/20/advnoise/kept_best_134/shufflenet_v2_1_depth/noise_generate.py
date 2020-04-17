import tensorflow as tf
import tensorflow.contrib.slim as slim



def noise_generator(x):

    with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, 
            activation_fn=tf.nn.relu):
        x = slim.conv2d(x, 32, (1, 1), stride=1, scope='Conv1_advnoise')
        x = slim.conv2d(x, 64, (1, 1), stride=1, scope='Conv2_advnoise')
        x = slim.conv2d(x, 3, (1, 1), stride=1, activation_fn=None,normalizer_fn=None,scope='Conv3_advnoise')
   
    return x


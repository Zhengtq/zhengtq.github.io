import tensorflow as tf
import tensorflow.contrib.slim as slim


BATCH_NORM_MOMENTUM = 0.997
BATCH_NORM_EPSILON = 1e-3


def inference(images, is_training=True, num_classes=2, depth_multiplier='1.0', reuse=None):


    possibilities = {'0.5': 48, '1.0': 116, '1.5': 176, '2.0': 224}
    initial_depth = possibilities[depth_multiplier]
    out_data = {}
    def batch_norm(x):
        x = tf.layers.batch_normalization(
            x, axis=3, center=True, scale=True,
            training=is_training,
            momentum=BATCH_NORM_MOMENTUM,
            epsilon=BATCH_NORM_EPSILON,
            fused=True, name='batch_norm'
        )
        return x

    with tf.name_scope('standardize_input'):
        x = images

    all_collect_map = []
    with tf.variable_scope('ShuffleNetV2', reuse=tf.AUTO_REUSE):
        params = {
            'padding': 'SAME', 'activation_fn': tf.nn.relu,
            'normalizer_fn': batch_norm, 'data_format': 'NHWC',
            'weights_initializer': tf.contrib.layers.xavier_initializer()
        }
        #wd=0.00018
        with slim.arg_scope([slim.conv2d, depthwise_conv], **params):
          with slim.arg_scope([slim.conv2d],
                  weights_regularizer=slim.l2_regularizer(0.000005)):


            x = slim.conv2d(x, 24, (3, 3), stride=2, scope='Conv1')
            x = slim.max_pool2d(x, (3, 3), stride=2, padding='SAME', scope='MaxPool')

            x = block(x, num_units=4, out_channels=initial_depth, scope='Stage2')
            all_collect_map.append(tf.image.resize_images(x, [64, 64]))
            x = block(x, num_units=8, scope='Stage3')
            all_collect_map.append(tf.image.resize_images(x, [64, 64]))
            x = block(x, num_units=4, scope='Stage4')
            all_collect_map.append(tf.image.resize_images(x, [64, 64]))

            final_channels = 1024 if depth_multiplier != '2.0' else 2048
            x = slim.conv2d(x, final_channels, (1, 1), stride=1, scope='Conv5')

            out_data['middle'] = x
    # global average pooling
#    x = tf.reduce_mean(x, axis=[1, 2])
            last_pool_kernel = int(x.get_shape()[-2]) 
            x = slim.avg_pool2d(x, [last_pool_kernel, last_pool_kernel]) 

            out_data['rnn_fea'] = x
#      logits = slim.fully_connected(
        #  x, num_classes, activation_fn=None, scope='Zoutput',
        #  weights_initializer=tf.contrib.layers.xavier_initializer()
    #  )
            x = slim.dropout(x, 0.3, is_training=is_training, scope='dropout6')
            logits = slim.conv2d(x, num_classes, [1,1],activation_fn=None, normalizer_fn=None,biases_initializer=tf.zeros_initializer(), scope='Conv2d_1c_1x1' )
            logits = tf.squeeze(logits, [1, 2])
            logits = tf.identity(logits, name='Zoutput')

            if is_training is True:
                out_data['depth'] = generate_depth_map(all_collect_map)
            else:
                out_data['depth'] = generate_depth_map(all_collect_map)  

            out_data['out_put'] = logits
    return logits, out_data



def generate_depth_map(all_collect_map):

    x = tf.concat(all_collect_map, axis=3)


    x = slim.conv2d(x, 128, (3, 3), stride=1,padding='SAME', scope='Conv1_gdepth')
    x = slim.conv2d(x, 64, (3, 3), stride=1,padding='SAME', scope='Conv2_gdepth')
    x = slim.conv2d(x, 1, (3, 3), stride=1, padding='SAME', normalizer_fn=None, activation_fn=None,scope='Conv3_gdepth')
   
    return x




def block(x, num_units, out_channels=None, scope='stage'):
    with tf.variable_scope(scope):

        with tf.variable_scope('unit_1'):
            x, y = basic_unit_with_downsampling(x, out_channels)

        for j in range(2, num_units + 1):
            with tf.variable_scope('unit_%d' % j):
                x, y = concat_shuffle_split(x, y)
                x = basic_unit(x)
        x = tf.concat([x, y], axis=3)

    return x


def concat_shuffle_split(x, y):
    with tf.name_scope('concat_shuffle_split'):
        shape = tf.shape(x)
        batch_size = shape[0]
        height, width = shape[1], shape[2]
        depth = x.shape[3].value

        z = tf.stack([x, y], axis=3)  # shape [batch_size, height, width, 2, depth]
        z = tf.transpose(z, [0, 1, 2, 4, 3])
        z = tf.reshape(z, [batch_size, height, width, 2*depth])
        x, y = tf.split(z, num_or_size_splits=2, axis=3)
        return x, y


def basic_unit(x):
    in_channels = x.shape[3].value
    x = slim.conv2d(x, in_channels, (1, 1), stride=1, scope='conv1x1_before')
    x = depthwise_conv(x, kernel=3, stride=1, activation_fn=None, scope='depthwise')
    x = slim.conv2d(x, in_channels, (1, 1), stride=1, scope='conv1x1_after')
    return x


def basic_unit_with_downsampling(x, out_channels=None):
    in_channels = x.shape[3].value
    out_channels = 2 * in_channels if out_channels is None else out_channels

    y = slim.conv2d(x, in_channels, (1, 1), stride=1, scope='conv1x1_before')
    y = depthwise_conv(y, kernel=3, stride=2, activation_fn=None, scope='depthwise')
    y = slim.conv2d(y, out_channels // 2, (1, 1), stride=1, scope='conv1x1_after')

    with tf.variable_scope('second_branch'):
        x = depthwise_conv(x, kernel=3, stride=2, activation_fn=None, scope='depthwise')
        x = slim.conv2d(x, out_channels // 2, (1, 1), stride=1, scope='conv1x1_after')
        return x, y


@tf.contrib.framework.add_arg_scope
def depthwise_conv(
        x, kernel=3, stride=1, padding='SAME',
        activation_fn=None, normalizer_fn=None,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        data_format='NHWC', scope='depthwise_conv'):

    with tf.variable_scope(scope):
        assert data_format == 'NHWC'
        in_channels = x.shape[3].value
        W = tf.get_variable(
            'depthwise_weights',
            [kernel, kernel, in_channels, 1], dtype=tf.float32,
            initializer=weights_initializer
        )
        x = tf.nn.depthwise_conv2d(x, W, [1, stride, stride, 1], padding, data_format='NHWC')
        x = normalizer_fn(x) if normalizer_fn is not None else x  # batch normalization
        x = activation_fn(x) if activation_fn is not None else x  # nonlinearity
        return x

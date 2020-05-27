import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from config import cfg
from se import Squeeze_excitation_layer as se_layer

def preact_conv(inputs, n_filters, filter_size=[3, 3], dropout_p=0.2):   
    preact = tf.nn.relu(slim.batch_norm(inputs))
    conv = slim.conv2d(preact, n_filters, filter_size, activation_fn=None, normalizer_fn=None)
    if dropout_p != 0.0:
      conv = slim.dropout(conv, keep_prob=(1.0-dropout_p))
    return conv

def DenseBlock(stack, n_layers, growth_rate, dropout_p, path_type, out_dim, scope=None):  
  with tf.name_scope(scope) as sc:
    new_features = []
    for j in range(n_layers):
      # Compute new feature maps
      layer = preact_conv(stack, growth_rate, dropout_p=dropout_p)
      new_features.append(layer)
      # stack new layer
      stack = tf.concat([stack, layer], axis=-1)
    new_features = tf.concat(new_features, axis=-1)

    # Special block for SE
    if path_type == "down":
      stack = se_layer(stack, out_dim, 1, (scope + path_type))
    else:
      new_features = se_layer(new_features, out_dim, 1, (scope + path_type))
    
    return stack, new_features


def TransitionLayer(inputs, n_filters, dropout_p=0.2, compression=1.0, scope=None):
  with tf.name_scope(scope) as sc:
    if compression < 1.0:
      n_filters = tf.to_int32(tf.floor(n_filters*compression))
    l = preact_conv(inputs, n_filters, filter_size=[1, 1], dropout_p=dropout_p)
    l = slim.pool(l, [2, 2], stride=[2, 2], pooling_type='AVG')
  return l

def TransitionDown(inputs, n_filters, dropout_p=0.2, scope=None):
  with tf.name_scope(scope) as sc:
    l = preact_conv(inputs, n_filters, filter_size=[1, 1], dropout_p=dropout_p)
    l = slim.pool(l, [2, 2], stride=[2, 2], pooling_type='MAX')
    return l

def TransitionUp(block_to_upsample, skip_connection, n_filters_keep, scope=None):
  with tf.name_scope(scope) as sc:
    # Upsample
    l = slim.conv2d_transpose(block_to_upsample, n_filters_keep, kernel_size=[3, 3], stride=[2, 2])
    # Concatenate with skip connection
    l = tf.concat([l, skip_connection], axis=-1)
    return l


def build_fc_densenet(inputs, preset_model='FC-DenseNet56', num_classes=2, n_filters_first_conv=48, n_pool=5, growth_rate=12, n_layers_per_block=4, dropout_p=0.2, scope=None):
    if preset_model == 'FC-DenseNet56':
      n_pool=5
      growth_rate=12
      n_layers_per_block=4
    elif preset_model == 'FC-DenseNet67':
      n_pool=5
      growth_rate=16
      n_layers_per_block=5
    elif preset_model == 'FC-DenseNet103':
      n_pool=5
      growth_rate=16
      n_layers_per_block=[4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]

    if type(n_layers_per_block) == list:
        assert (len(n_layers_per_block) == 2 * n_pool + 1)
    elif type(n_layers_per_block) == int:
        n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
    else:
        raise ValueError

    with tf.variable_scope(scope, preset_model, [inputs]) as sc:

      # We perform a first convolution.
      stack = slim.conv2d(inputs, n_filters_first_conv, [3, 3], scope='first_conv')

      n_filters = n_filters_first_conv

      # Downsampling path
      skip_connection_list = []

      for i in range(n_pool):
        n_filters += growth_rate * n_layers_per_block[i]
        # Dense Block
        stack, _ = DenseBlock(stack, n_layers_per_block[i], growth_rate, dropout_p, "down", n_filters, scope='denseblock%d' % (i+1))
        
        # At the end of the dense block, the current stack is stored in the skip_connections list
        skip_connection_list.append(stack)

        # Transition Down
        stack = TransitionDown(stack, n_filters, dropout_p, scope='transitiondown%d'%(i+1))

      skip_connection_list = skip_connection_list[::-1]

      #Bottleneck Dense Block
      out_dim = n_layers_per_block[n_pool]*growth_rate
      # We will only upsample the new feature maps
      stack, block_to_upsample = DenseBlock(stack, n_layers_per_block[n_pool], growth_rate, dropout_p, "bottlneck", out_dim, scope='denseblock%d' % (n_pool + 1))

      # Upsampling path
      for i in range(n_pool):
        # Transition Up ( Upsampling + concatenation with the skip connection)
        n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
        stack = TransitionUp(block_to_upsample, skip_connection_list[i], n_filters_keep, scope='transitionup%d' % (n_pool + i + 1))

        # Dense Block
        out_dim =  n_layers_per_block[n_pool + i + 1]*growth_rate
        # We will only upsample the new feature maps
        stack, block_to_upsample = DenseBlock(stack, n_layers_per_block[n_pool + i + 1], growth_rate, dropout_p, "up", out_dim, scope='denseblock%d' % (n_pool + i + 2))

      # Softmax
      net = slim.conv2d(stack, num_classes, [1, 1], scope='logits')
      prediction = tf.argmax(net, axis=3, output_type=tf.int32)
      probabilities = tf.nn.softmax(net)
      
      return net, prediction, probabilities
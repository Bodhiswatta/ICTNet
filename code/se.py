import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
#from tflearn.layers.conv import global_avg_pool
import numpy as np

from config import cfg

#os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def Global_Average_Pooling(x):
  #return global_avg_pool(x, name='Global_avg_pooling')
  return tf.reduce_mean(x, axis=[1,2])

def Fully_connected(x, units, layer_name='fully_connected') :
  with tf.name_scope(layer_name) :
    return tf.layers.dense(inputs=x, use_bias=True, units=units)

def Relu(x):
  return tf.nn.relu(x)

def Sigmoid(x):
  return tf.nn.sigmoid(x)

def Squeeze_excitation_layer(input_x, out_dim, ratio, layer_name):
  with tf.name_scope(layer_name) :
    squeeze = Global_Average_Pooling(input_x)

    excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
    excitation = Relu(excitation)
    excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
    excitation = Sigmoid(excitation)

    excitation = tf.reshape(excitation, [-1,1,1,out_dim])
    scale = input_x * excitation

  return scale
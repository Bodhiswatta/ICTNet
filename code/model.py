import os
import tensorflow as tf

from config import cfg

class Model(object):
    def __init__(self, img_channels=3, num_label=2, call_type='training'):
        '''
        Args:
            img_channels: Integer, the channels of input.
            num_label: Integer, the category number.
        '''
        self.batch_size = cfg.batch_size
        if call_type == 'training':
            self.height = cfg.patch_size
            self.width = cfg.patch_size
        else:
            self.height = cfg.test_patch_size
            self.width = cfg.test_patch_size
        self.img_channels = img_channels
        self.num_label = num_label

        from ictnet import build_fc_densenet as build_model

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.image = tf.placeholder(tf.float32, shape=(self.batch_size, self.height, self.width, self.img_channels))
            self.labels = tf.placeholder(tf.uint8, shape=(self.batch_size, self.height, self.width, 1))
            if cfg.is_training and call_type=='training':
                if cfg.is_one_hot:
                    self.mask = tf.one_hot(self.labels, depth=self.num_label, axis=-1, dtype=tf.float32)
                else:
                    self.mask = self.labels
                print("Model creation start.")
                self.network, self.prediction, self.probabilities = build_model(self.image, num_classes=self.num_label, preset_model='FC-DenseNet103', dropout_p=0.0)
                print("Model creation complete.")

                self.loss()
                self._train_summary()
                self.global_step = tf.Variable(1, name='global_step', trainable=False)

                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.995)
                self.train_op = self.optimizer.minimize(self.loss, var_list=[var for var in tf.trainable_variables()], global_step=self.global_step)
            else:
                self.network, self.prediction, self.probabilities = build_model(self.image, num_classes=self.num_label, preset_model='FC-DenseNet103', dropout_p=0.0)

    def loss(self):        
        # 1. Cross-entropy loss
        self.cross_entropy_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.network, labels=self.mask))

        # 2. Total loss
        self.loss = self.cross_entropy_loss

        # Accuracy
        correct_prediction = tf.equal(tf.to_int32(self.labels), tf.reshape(self.prediction, shape=[self.batch_size, self.height, self.width, 1]))
        self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    # Summary
    def _train_summary(self):
        train_summary = []
        train_summary.append(tf.summary.scalar('train/cross_entropy_loss', self.loss))
        train_summary.append(tf.summary.scalar('train/accuracy', self.accuracy))
        
        train_summary.append(tf.summary.image('train/original', self.image))
        train_summary.append(tf.summary.image('train/ground_truth', tf.to_float(self.labels)))
        train_summary.append(tf.summary.image('train/building_prob', tf.reshape(self.probabilities[:, :, :, 1], shape=[self.batch_size, self.height, self.width, 1])))
        train_summary.append(tf.summary.image('train/building_pred', tf.reshape(tf.to_float(self.prediction), shape=[self.batch_size, self.height, self.width, 1])))
                
        #train_summary.append(tf.summary.histogram('train/activation', self.activation))
        self.train_summary = tf.summary.merge(train_summary)
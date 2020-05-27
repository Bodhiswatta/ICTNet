import os
import tensorflow as tf

flags = tf.app.flags


############################
#    hyper parameters      #
############################

flags.DEFINE_integer('batch_size', 4, 'batch size')
flags.DEFINE_integer('patch_size', 256, 'the input train patch size')
flags.DEFINE_integer('test_patch_size', 768, 'the input test/validation patch size')
flags.DEFINE_integer('num_labels', 2, 'the number of labels')
flags.DEFINE_integer('img_channels', 3, 'the number of channels for input image')
flags.DEFINE_integer('evaluate_every_n_epoch', 1, 'number of epoch of training after which evaluation is run on validation data')
flags.DEFINE_string('patch_generation', 'alternating', 'patch generation technique [sequential, random, alternating]')

flags.DEFINE_integer('train_tile_padding', 60, 'number of pixels padded to each side of the tile')
flags.DEFINE_integer('test_tile_padding', 188, 'number of pixels padded to each side of the tile at test time')
flags.DEFINE_float('threshold', 0.4, 'the threshold value used for converting the probability map into a binary mask')
flags.DEFINE_boolean('per_tile_validation_accuracy', True, 'True: get validation accuracy for every tile. False: get overall validation accuracy')
flags.DEFINE_boolean('test_time_augmentation', False, 'True: check and generate(if required) TTA tiles for test/validation. False: get overall validation accuracy')
flags.DEFINE_string('gpu_number', '0', 'which GPU to use for running the network')
flags.DEFINE_boolean('activate_tensorboard', False, 'True: Training progress can be tracked using tensorboard. False: tensorboard logging is deactivated')
flags.DEFINE_string('data_path', '../data', 'Data directory. [Default value: ../data]')

flags.DEFINE_boolean('GeoTIFF_output', True, 'True: output raster is converted to GeoTiff. False: output raster is a Tiff file.')
flags.DEFINE_string('output_shape_type', 'ESRI Shapefile', 'output shape type options [ESRI Shapefile, GeoJSON]')
flags.DEFINE_integer('min_building_area', 30, 'Number of pixels which is the minimum size of buildings considered during postprocessing')
flags.DEFINE_float('tolerance', 0.5, 'the tolerance value used for smoothing the shape files')

flags.DEFINE_integer('epoch', 200, 'epoch')
flags.DEFINE_integer('save_freq', 1, 'the frequency of saving model(epoch)')

flags.DEFINE_boolean('is_training', False, 'True: Train phase. False: Predict phase.')
flags.DEFINE_boolean('data_augmentation', True, 'apply data augmentation')
flags.DEFINE_boolean('scale_augmentation', True, 'apply random multi-scale data augmentation [valid only when data_augmentation is True]')
flags.DEFINE_boolean('rotation_flip_augmentation', True, 'apply random rotation and flip data augmentation [valid only when data_augmentation is True]')
flags.DEFINE_boolean('contrast_brightness_augmentation', False, 'apply contrast perturbation as data augmentation [valid only when data_augmentation is True]')

flags.DEFINE_boolean('compute_mean_rgb', False, 'True: compute mean RGB values of training images. False: use default values')
flags.DEFINE_boolean('maintain_ratio', False, 'True: maintain building to non-building ratio of set thresholds for first 5 epochs. False: use all data for training')
flags.DEFINE_integer('image_tile_per_mini_epoch', 5, 'the number of image tiles used for training 1 mini epoch [change the number depending on memory capacity]')

results = os.path.join('..', 'results')
logdir = os.path.join(results, 'logdir')
flags.DEFINE_string('results', results, 'path for saving results')
flags.DEFINE_string('logdir', logdir, 'logs directory')
flags.DEFINE_boolean('debug', False, 'debug mode')
summarydir = os.path.join('results', 'summary')
flags.DEFINE_string('summarydir', summarydir, 'Summary directory')
flags.DEFINE_boolean('is_one_hot', True, 'mask need one hot encoding')

cfg = tf.app.flags.FLAGS

# Uncomment this line to run in debug mode
#tf.logging.set_verbosity(tf.logging.INFO)

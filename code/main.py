import time
import os
from os import path
#########
import logging
logging.basicConfig(filename='../log/tensorflow.log',level=logging.DEBUG)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
#########
from config import cfg
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_number

import sys
import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm

sys.path.append('..')
sys.path.append('.')

from model import Model

from data_loader import load_training_data, get_image_path, get_image_patches, get_patch_weights, mean_r, mean_g, mean_b
from postProcessing import merge_TTA_tiles_by_averagring, apply_threshold
from compute_accuracy import calculate_acc
from raster2shape_gdal import convert_raster_to_geoTiff_dir, raster_to_shape_dir

data_path = cfg.data_path

def train(current_epoch, global_step, x=1):
    model = Model(img_channels=cfg.img_channels, num_label=cfg.num_labels)
    tf.logging.info(' Graph loaded')
    
    with tf.Session(graph=model.graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver = tf.train.Saver(max_to_keep=2, save_relative_paths=True)
        if str(tf.train.latest_checkpoint(cfg.logdir)) == 'None':
            init = tf.initialize_all_variables()
            sess.run(init)
            print('No model weights to restore')
            tf.logging.info('No model to restore!')
        else:
            saver.restore(sess,tf.train.latest_checkpoint(cfg.logdir))
            print('Model weights Restored')
            tf.logging.info('Model restored!')
        train_writer = tf.summary.FileWriter(cfg.logdir + '/train', sess.graph)
                
        for epoch in range(x):
            total_num_tr_batch = 0
            data_loader = load_training_data(current_epoch)
            num_calls = next(data_loader)
            print('Training for epoch ' + str(current_epoch + 1) + ': ')
            for mini_epoch in range(num_calls):
                print('mini_epoch #: {0}'.format(mini_epoch + 1))
                trX, trY, num_tr_batch, valX, valY, num_val_batch = next(data_loader)
                total_num_tr_batch += num_tr_batch
                avg_loss = []
                avg_acc = []
                for step in tqdm(range(num_tr_batch), total=num_tr_batch, ncols=100, leave=False, unit='b'):
                    start = step * cfg.batch_size
                    end = start + cfg.batch_size
                    global_step += 1

                    _, loss, train_acc, summary_str = sess.run([model.train_op, model.loss, model.accuracy, model.train_summary], {model.image: trX[start:end], model.labels: trY[start:end]})
                    assert not np.isnan(loss), 'Something wrong! loss is nan...'
                    if cfg.activate_tensorboard:
                        train_writer.add_summary(summary_str, global_step)
                    
            if (epoch + 1) % cfg.save_freq == 0:
                saver.save(sess, cfg.logdir + '/model_V1_epoch_%04d_step_%02d' % (epoch+1, global_step))
        
        return global_step

def evaluation(epoch =-1, eval_type = 'training'):
    model = Model(img_channels=cfg.img_channels, num_label=cfg.num_labels, call_type = 'evaluation')
    tf.logging.info(' Graph loaded')

    threshold = cfg.threshold
    if cfg.is_training:
        in_img_path = data_path + '/validation/images/'
        path = data_path + '/validation/results'
    else:
        in_img_path = data_path + '/test/images/'
        path = data_path + '/test/results'

    max = 0
    for file in os.listdir(path):
        if int(file) > max:
            max = int(file)
    max += 1
    os.makedirs(path + '/' + str(max) + '/raw')
    os.makedirs(path + '/' + str(max) + '/final')
    os.makedirs(path + '/' + str(max) + '/merged')
    os.makedirs(path + '/' + str(max) + '/shape')
    os.makedirs(path + '/' + str(max) + '/final_geoTiff')
    out_img_path = path + '/' + str(max)

    patch_weights = get_patch_weights()

    with tf.Session(graph=model.graph) as sess:
        saver = tf.train.Saver()
        if str(tf.train.latest_checkpoint(cfg.logdir)) != 'None':
            saver.restore(sess,tf.train.latest_checkpoint(cfg.logdir))
            tf.logging.info('Model restored!')
            print('Model restored!')
        else:
            tf.logging.error('Evaluation failed. No model to Restore')
            print('Evaluation failed. No model to Restore')
            return
        #teY: dummy variable to be passed to model
        teY = np.zeros((cfg.batch_size, cfg.test_patch_size, cfg.test_patch_size, 1), dtype=np.uint8)
        
        #get data for evaluation
        image_path = get_image_path()
        num_eval_files = next(image_path)
        for i in range(num_eval_files):
            file_path = next(image_path)
            file_name = file_path.split('/')[-1]
            stride_val = cfg.test_patch_size//2            
            image_patches = get_image_patches(file_path, batch_size=cfg.batch_size, patch_size=cfg.test_patch_size, stride=stride_val)
            x_len, y_len = next(image_patches)
            call_count = next(image_patches)
            print('No of calls for image ' + file_name  + ' : ' + str(call_count))
            out_img = np.zeros((x_len, y_len))
            pred_img = np.zeros((x_len, y_len))
            for j in range(call_count):
                print('Call #: ' + str(j))
                img_patch_array = next(image_patches)
                img_patch_position_list = next(image_patches)
                counter = next(image_patches)
                
                num_te_batch = int(counter/cfg.batch_size)
                teX = img_patch_array
                for i in tqdm(range(num_te_batch), total=num_te_batch, ncols=100, leave=False, unit='b'):
                    start = i * cfg.batch_size
                    end = start + cfg.batch_size
                    out_prediction, out_probabilities = sess.run([model.prediction, model.probabilities], {model.image: teX[start:end], model.labels: teY})
                    out_images = np.array(out_probabilities.squeeze()[:, :, :, 1])
                    out_prediction = np.array(out_prediction.squeeze()).astype(np.float32)
                    for j in range(cfg.batch_size):
                        if start+j < len(img_patch_position_list):
                            eval_img = np.array(out_images[j])
                            eval_img = eval_img.astype(np.float32)                        
                            start_x, start_y = img_patch_position_list[start+j]
                            out_img[start_x:start_x+cfg.test_patch_size, start_y:start_y+cfg.test_patch_size] += eval_img * patch_weights
                            eval_pred = np.array(out_prediction[j])
                            pred_img[start_x:start_x+cfg.test_patch_size, start_y:start_y+cfg.test_patch_size] += eval_pred * patch_weights
                        
            img_map = next(image_patches)            
            output_img = out_img/img_map
            output_img = output_img[cfg.test_tile_padding:-cfg.test_tile_padding, cfg.test_tile_padding:-cfg.test_tile_padding]
            cv2.imwrite(out_img_path + '/raw/' + file_name, (output_img*255.).astype(np.uint8))            
            print('Evaluation results for ' + file_name + ' saved to: ' + out_img_path + '/raw/' + file_name)
        
        raw_path = out_img_path + '/raw/'
        merged_path = out_img_path + '/merged/'
        final_path = out_img_path + '/final/'
        final_geotiff_path = out_img_path + '/final_geoTiff/'
        shape_path = out_img_path + '/shape/'
        if cfg.test_time_augmentation:            
            merge_TTA_tiles_by_averagring(raw_path, merged_path)
            apply_threshold(merged_path, final_path, threshold)
        else:
            apply_threshold(raw_path, final_path, threshold)
        
        if cfg.GeoTIFF_output:
            convert_raster_to_geoTiff_dir(in_img_path, final_path, final_geotiff_path)
            raster_to_shape_dir(final_geotiff_path, shape_path)

        if eval_type == 'training':
            print('Begin Evaluation..!!')
            result = calculate_acc(final_path)
            print("#####################\n" + result + "#####################\n")
            final = "folder: {0}\nepoch: {1}, patch_size: {2}, batch_size: {3}, dataAug: {4}".format(final_path, epoch + 1, cfg.test_patch_size, cfg.batch_size, cfg.data_augmentation)
            final += ("\n#####################\n" + result + "#####################\n")
            with open("../eval.txt", "a") as evalfile:
                evalfile.write(final)
            print('End of Evaluation..!!')

def main(_):
    global_step = 0
    cfg.is_training = False
    if cfg.is_training:
        for epoch in range(cfg.epoch):
            global_step = train(epoch, global_step)
            if epoch % cfg.evaluate_every_n_epoch == 0:
                evaluation(epoch)
    else:
        evaluation(eval_type = 'evaluation')

if __name__ == "__main__":
    tf.app.run()

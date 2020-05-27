import os
import cv2
import numpy as np
import tensorflow as tf
import random
import math

from config import cfg
from data_augmentation import get_training_patches, check_for_TTA, generate_TTA_tiles

ratio_threshold = [0.25, 0.2, 0.15, 0.1, 0.05]
data_path = cfg.data_path
image_folder = 'images'
gt_folder = 'ground_truth'
log_path = '../log'
mean_r, mean_g, mean_b = -1, -1, -1

def get_mean_color_values():
    print("Calculating mean colors for training dataset!!")
    img_path = data_path + '/train/' + image_folder + '/'
    r_channels = []
    g_channels = []
    b_channels = []
    for file in os.listdir(img_path):
        img = cv2.imread(img_path + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        r_channels.append(img[:,:,0])
        g_channels.append(img[:,:,1])
        b_channels.append(img[:,:,2])
    mean_r = np.mean(np.array(r_channels))
    mean_g = np.mean(np.array(g_channels))
    mean_b = np.mean(np.array(b_channels))
    print("mean values for - R: {0}, G: {1}, B: {2}".format(mean_r, mean_g, mean_b))
    return mean_r, mean_g, mean_b

if cfg.compute_mean_rgb and mean_r == -1 and mean_g == -1 and mean_b == -1:
    mean_r, mean_g, mean_b = get_mean_color_values()
else:
    mean_r, mean_g, mean_b = 103.61419795694658, 109.0776216765373, 100.39708955555578

def get_num_patches_per_tile(height, width, training_stride=cfg.patch_size//2):
    total_width = width + (2 * cfg.train_tile_padding)
    num_patches_width = int((total_width-cfg.patch_size)/training_stride)
    total_height = height + (2 * cfg.train_tile_padding)
    num_patches_height = int((total_height-cfg.patch_size)/training_stride)
    return num_patches_height, num_patches_width

def load_training_data(epoch):
    batch_size = cfg.batch_size
    patch_size = cfg.patch_size
    path = data_path
    img_path = data_path + '/train/' + image_folder + '/'
    gt_path = data_path + '/train/' + gt_folder + '/'

    num_building_px = 0
    total_num_px = 0

    image_files = os.listdir(img_path)
    random.shuffle(image_files)

    if len(image_files) % cfg.image_tile_per_mini_epoch == 0:
        call_count = len(image_files) // cfg.image_tile_per_mini_epoch
    else:
        call_count = (len(image_files) // cfg.image_tile_per_mini_epoch) + 1
    #call_count is the number of times the loop will iterate to get all patches for 1 epoch
    yield call_count

    height, width, _ = cv2.imread(img_path + image_files[0]).shape

    stride = cfg.patch_size//2
    num_patches_height, num_patches_width = get_num_patches_per_tile(height, width, stride)
    num_patches_per_tile = num_patches_height * num_patches_width
    train_count = num_patches_per_tile * cfg.image_tile_per_mini_epoch
    train_count -= train_count % cfg.batch_size
    val_count = 10 * cfg.batch_size
    
    num_patches_per_img = num_patches_per_tile
    x_list = []
    y_list = []
    if cfg.patch_generation == 'sequential' or (cfg.patch_generation == 'alternating' and epoch%2==0):
        for i in range(num_patches_width):
            x_list += [i * stride]
        x_list *= num_patches_height
        for i in range(num_patches_height):
            y_list += [i * stride] * num_patches_width
        ########### to shuffle the list of x and corresponding y positions ############
        shuffled_index = random.sample(range(0,len(x_list)), len(x_list))
        x_list_new = []
        y_list_new = []
        for position_index in shuffled_index:
            x_list_new.append(x_list[position_index])
            y_list_new.append(y_list[position_index])
        x_list = x_list_new
        y_list = y_list_new
    else:
        # x_list, y_list = np.random.randint(cfg.tile_width + (2 * cfg.tile_padding)-cfg.patch_size, size = (2, num_patches_per_img))
        x_list = np.random.randint(width + (2 * cfg.train_tile_padding)-cfg.patch_size, size = (num_patches_per_img))
        y_list = np.random.randint(height + (2 * cfg.train_tile_padding)-cfg.patch_size, size = (num_patches_per_img))

    for call_num in range(call_count):
        img_tiles = []
        gt_tiles = []
        for i in range(cfg.image_tile_per_mini_epoch):
            if len(image_files) == 0:
                break
            image_file = image_files.pop()
            img = cv2.imread(img_path + image_file)
            img = cv2.copyMakeBorder(img, cfg.train_tile_padding, cfg.train_tile_padding, cfg.train_tile_padding, cfg.train_tile_padding, cv2.BORDER_REFLECT_101)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tiles.append(img)
            gt = cv2.imread(gt_path + image_file, 0)
            gt = cv2.copyMakeBorder(gt, cfg.train_tile_padding, cfg.train_tile_padding, cfg.train_tile_padding, cfg.train_tile_padding, cv2.BORDER_REFLECT_101)
            gt_tiles.append(gt)
        
        images, ground_truths = get_training_patches(img_tiles, gt_tiles, x_list, y_list, mean_r, mean_g, mean_b)
        
        images = np.array(images).astype(np.float32) / 255.
        ground_truths = np.array(ground_truths).astype(np.float32) / 255.

        # else block drops batches if building to non-building ratio is less than than threshold for first 5 epoch and maintain_ratio flag is set to True
        if not cfg.maintain_ratio or epoch > 4:
            trX = images[:train_count]
            trY = ground_truths[:train_count]

            valX = images[-val_count:]
            valY = ground_truths[-val_count:]

            trY = trY.reshape((-1, cfg.patch_size, cfg.patch_size, 1))
            valY = valY.reshape((val_count, cfg.patch_size, cfg.patch_size, 1))

            num_tr_batch = trY.shape[0] // cfg.batch_size
            num_val_batch = val_count // cfg.batch_size
        else:
            threshold = ratio_threshold[epoch]
            special_batch_images = []
            special_batch_gt = []
            for batch_index in range(0, images.shape[0], cfg.batch_size):
                check_batch = np.array(ground_truths[batch_index:batch_index+cfg.batch_size]).astype(np.uint8)
                building_px_count = np.count_nonzero(check_batch)
                total_px_count = check_batch.size                
                if building_px_count/total_px_count >= threshold:
                    # for batch in range(cfg.batch_size):
                    special_batch_images.extend(images[batch_index : batch_index+cfg.batch_size])
                    special_batch_gt.extend(ground_truths[batch_index : batch_index+cfg.batch_size])
            special_train_count = len(special_batch_images)
            special_val_count = 2 * cfg.batch_size
            images = np.array(special_batch_images)
            ground_truths = np.array(special_batch_gt)
            
            trX = images[:special_train_count]
            trY = ground_truths[:special_train_count]

            valX = images[-special_val_count:]
            valY = ground_truths[-special_val_count:]

            trY = trY.reshape((special_train_count, cfg.patch_size, cfg.patch_size, 1))
            valY = valY.reshape((special_val_count, cfg.patch_size, cfg.patch_size, 1))

            num_tr_batch = special_train_count // cfg.batch_size
            num_val_batch = special_val_count // cfg.batch_size
        
        yield trX, trY.astype(np.uint8), num_tr_batch, valX, valY.astype(np.uint8), num_val_batch
        
        num_building_px += np.count_nonzero(trY)
        total_num_px += trY.size
    building_total_ratio = round((num_building_px / total_num_px) * 100, 2)
    print("epoch: {0}, building to total pixels ratio: {1}%".format(epoch, building_total_ratio))

def get_patch_weights(patch_size=cfg.test_patch_size):
    choice = 1
    if choice == 0:
        step_size = (1.0 - 0.5)/(patch_size/2)
        a = np.arange(1.0, 0.5, -step_size)
        b = a[::-1]
        c = np.concatenate((b,a))
        ct = c.reshape(-1,1)
        x = ct*c
        return x
    elif choice == 1:
        min_weight = 0.5
        step_count = patch_size//4
        step_size = (1.0 - min_weight)/step_count
        a = np.ones(shape=(patch_size,patch_size), dtype=np.float32)
        a = a * min_weight        
        for i in range(1, step_count + 1):
            a[i:-i, i:-i] += step_size
        a = cv2.GaussianBlur(a,(5,5),0)
        return a
    else:
        a = np.ones(shape=(patch_size,patch_size), dtype=np.float32)
        return a

def get_image_path():
    if cfg.is_training:
        test_img_path = data_path + '/validation/' + image_folder + '/'
    else:
        test_img_path = data_path + '/test/' + image_folder + '/'
    
    tta_status = check_for_TTA(test_img_path)
    if cfg.test_time_augmentation and not tta_status:
        generate_TTA_tiles(test_img_path, test_img_path)
    
    # files = [f for f in os.listdir(test_img_path) if f.endswith('.tif')]    
    files = [f for f in os.listdir(test_img_path)]
    file_count = len(files)
    yield file_count
    for fi in files:
        yield str(test_img_path + '/' + fi)

def get_image_patches(img_path, patch_size=cfg.test_patch_size, batch_size=cfg.batch_size, batch_count=250, stride=cfg.test_patch_size//2, log_path='../log'):
    img = cv2.imread(img_path)
    img = cv2.copyMakeBorder(img, cfg.test_tile_padding, cfg.test_tile_padding, cfg.test_tile_padding, cfg.test_tile_padding, cv2.BORDER_REFLECT_101)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    ##Segment to get mean subtracted image
    mean_adjusted_img_r = img[:,:,0] - mean_r
    mean_adjusted_img_g = img[:,:,1] - mean_g
    mean_adjusted_img_b = img[:,:,2] - mean_b
    img = cv2.merge([mean_adjusted_img_r, mean_adjusted_img_g, mean_adjusted_img_b]).astype(np.float32)
    img = img / 255.
    x, y, z = img.shape    
    yield x, y
    call_count = math.ceil((((x - patch_size)/stride + 1)*((y - patch_size)/stride + 1))/(batch_size*batch_count))
    yield call_count
    # print('call_count: {0}'.format(call_count))
    img_map = np.zeros((x, y), dtype=np.float32)
    patch_weights = get_patch_weights()
    counter = 0
    img_patch_array = []
    img_patch_position_list = []
    img_patch_position_list_total = []
    for i in range(0, x-patch_size+stride, stride):
        if i >= (x - patch_size):
            x_start = x - patch_size
        else:
            x_start = i
        for j in range(0, y-patch_size+stride, stride):
            if j >= (y - patch_size):
                y_start = y - patch_size
            else:
                y_start = j
            img_patch_array.append(img[x_start:x_start+patch_size,y_start:y_start+patch_size,:])
            img_patch_position_list.append((x_start, y_start))
            #img_map[x_start:x_start+patch_size,y_start:y_start+patch_size] += 1
            img_map[x_start:x_start+patch_size,y_start:y_start+patch_size] += patch_weights
            counter += 1
            if counter%(batch_size*batch_count) == 0:
                yield np.array(img_patch_array)
                yield img_patch_position_list
                yield counter
                img_patch_position_list_total.append(img_patch_position_list)
                counter = 0
                img_patch_array = []
                img_patch_position_list = []
    if counter > 0:
        if counter%batch_size > 0:
            for i in range(batch_size - (counter%batch_size)):
                img_patch_array.append(np.zeros((patch_size, patch_size, 3)))
                counter += 1
        yield np.array(img_patch_array)
        yield img_patch_position_list
        yield counter
        img_patch_position_list_total.append(img_patch_position_list)
    yield img_map

import cv2
import numpy as np
import tensorflow as tf
import os

from config import cfg

smallest_patch_size = (cfg.patch_size // 4) * 3
largest_patch_size = (cfg.patch_size // 2) * 3 + 1

def get_training_patches(images, ground_truths, x_list, y_list, mean_r, mean_g, mean_b):
    img_patches = []
    gt_patches = []
    for img_num in range(len(images)):
        img = images[img_num]
        gt = ground_truths[img_num]
        ####### rotation and flip augmentation [start] #######
        if cfg.data_augmentation and cfg.rotation_flip_augmentation:
            option = np.random.randint(8)
            # print("rotation and flip augmentation option: {0}".format(option))
            if option < 6:
                img, gt = rotate_flip_image(img, gt)
        ####### rotation and flip augmentation [end] #######
        ####### contrast and brightness augmentation [start] #######
        if cfg.data_augmentation and cfg.contrast_brightness_augmentation:
            option = np.random.randint(10)
            # print("contrast and brightness augmentation option: {0}".format(option))
            if option < 7:
                img = change_contrast_brightness(img)
        ######## contrast and brightness augmentation [end] #######
        ######## subtract mean r,g,b value for whole training dataset [starts] ########
        img = img.astype(np.float32)
        mean_adjusted_img_r = img[:,:,0] - mean_r
        mean_adjusted_img_g = img[:,:,1] - mean_g
        mean_adjusted_img_b = img[:,:,2] - mean_b
        img = cv2.merge([mean_adjusted_img_r, mean_adjusted_img_g, mean_adjusted_img_b])
        ######## subtract mean r,g,b value for whole training dataset [end] ########
        for patch_num in range(len(x_list)):
            x = x_list[patch_num]
            y = y_list[patch_num]
            if not cfg.data_augmentation or (cfg.data_augmentation and not cfg.scale_augmentation):        
                img_patch = img[x:x+cfg.patch_size,y:y+cfg.patch_size, :]
                gt_patch = gt[x:x+cfg.patch_size,y:y+cfg.patch_size]
            else:
                ####### scale augmentation [start] #######
                option = np.random.randint(3)
                if option < 2:
                    patch_size = cfg.patch_size
                else:
                    patch_size = np.random.randint(smallest_patch_size, largest_patch_size)
                    # print("Special patch size is: {0}".format(patch_size))
                if x + patch_size > img.shape[0]:
                    x -= (x + patch_size - img.shape[0])
                if y + patch_size > img.shape[1]:
                    y -= (y + patch_size - img.shape[1])
                img_patch = img[x:x+patch_size, y:y+patch_size, :]
                gt_patch = gt[x:x+patch_size, y:y+patch_size]
                if option > 0:
                    img_patch, gt_patch = resize_image(img_patch, gt_patch)
                ####### scale augmentation [end] #######
            img_patches.append(img_patch)
            gt_patches.append(gt_patch)
    return img_patches, gt_patches

def resize_image(img, gt):
    new_img = cv2.resize(img, (cfg.patch_size, cfg.patch_size), interpolation = cv2.INTER_CUBIC)
    new_gt = cv2.resize(gt, (cfg.patch_size, cfg.patch_size), interpolation = cv2.INTER_CUBIC)
    return new_img, new_gt

def rotate_flip_image(img, gt):
    k = np.random.randint(6)    #[k = 0-2:rotate random between 0-360degree, 3: vertical-flip, 4: horizontal-flip, 5: both-flip]
    ###### Image rotation ######
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    angle = np.random.randint(0, 360)
    scale = 1.0
    if k < 3:
        M = cv2.getRotationMatrix2D(center, angle, scale)
        new_img = cv2.warpAffine(img, M, (h, w), borderMode=cv2.BORDER_REFLECT101)
        new_gt = cv2.warpAffine(gt, M, (h, w), borderMode=cv2.BORDER_REFLECT101)
    ###### Image rotation ######
    ###### Image flip ######
    if k == 3:
        flip_axis = 0
    elif k == 4:
        flip_axis = 1
    elif k == 5:
        flip_axis = -1
    if k > 2 and k < 6:
        new_img = cv2.flip(img, flip_axis)
        new_gt = cv2.flip(gt, flip_axis)
    ###### Image flip ######
    return new_img, new_gt

def change_contrast_brightness(img):
    choice = np.random.randint(4)
    if choice == 0:
        updated_img = blur_image(img)
    elif choice == 1:
        updated_img = add_gaussian_noise(img)
    elif choice == 2:
        updated_img = clahe(img)
    else:
        updated_img = augment_contrast_brightness(img)
    return updated_img

def blur_image(img):
    choice = np.random.randint(2)
    if choice == 0:
        blur_img = cv2.GaussianBlur(img,(5,5),0)
    else:
        blur_img = cv2.medianBlur(img, 5)
    return blur_img


def add_gaussian_noise(img):
    mean = np.mean(img)
    fixed_sd = 15
    noisy_img = img + np.random.normal(mean, fixed_sd, img.shape).astype(np.int32)
    noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img_clipped

def clahe(img):
    gridsize = 100    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr

def augment_contrast_brightness(img):
    img = img.astype(np.float32)
    contrast = np.random.randint(980, 1021)/1000.
    brightness = np.random.randint(-2, 3)
    # print("contrast: {0}, brightness: {1}".format(contrast, brightness))
    alpha = contrast
    beta = 0
    gamma = brightness
    updated_img = cv2.addWeighted( img, contrast, img, 0, brightness)
    updated_img = np.clip(updated_img, 0, 255)
    return updated_img.astype(np.uint8)


def TTA_tiles(img):
    img_90 = np.rot90(img, 1).astype(np.uint8)
    img_180 = np.rot90(img, 2).astype(np.uint8)
    img_270 = np.rot90(img, 3).astype(np.uint8)
    
    V_flip_axis = 0
    H_flip_axis = 1
    img_V_flip = cv2.flip(img, V_flip_axis).astype(np.uint8)
    img_H_flip = cv2.flip(img, H_flip_axis).astype(np.uint8)

    img_clahe = clahe(img).astype(np.uint8)

    return img_90, img_180, img_270, img_V_flip, img_H_flip, img_clahe

def generate_TTA_tiles(in_path, out_path):
    for file in os.listdir(in_path):
        print("generating TTA tiles for: " + file)
        img = cv2.imread(in_path + file)
        img_90, img_180, img_270, img_V_flip, img_H_flip, img_clahe = TTA_tiles(img)
        name = file.split('.')[0]
        extn = file.split('.')[1]
        cv2.imwrite(out_path + name + '-90.' + extn, img_90)
        cv2.imwrite(out_path + name + '-180.' + extn, img_180)
        cv2.imwrite(out_path + name + '-270.' + extn, img_270)
        cv2.imwrite(out_path + name + '-V.' + extn, img_V_flip)
        cv2.imwrite(out_path + name + '-H.' + extn, img_H_flip)
        cv2.imwrite(out_path + name + '-clahe.' + extn, img_clahe)

def check_for_TTA(in_path):
    print("Checking for TTA")
    flag = False
    tta_type = '-clahe'
    extn = '.tif'
    files = os.listdir(in_path)
    file_names = [x.replace(tta_type + extn,'') for x in files if '.' in x and tta_type in x]
    # print(file_names)
    counter = 0
    for file_name in file_names:
        if sum(file_name in file for file in files) == 7:
            counter += 1
    if counter > 0 and counter == len(file_names):
        flag = True
    print("TTA status: {0}".format(flag))
    return flag
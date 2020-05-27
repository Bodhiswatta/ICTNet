import numpy as np
import cv2
import os
from os import path

def merge_TTA_tiles_by_averagring(in_path, out_path):
    extn_list = [' ', '-90', '-180', '-270', '-V', '-H', '-clahe']
    file_type = '.tif'
    files = os.listdir(in_path)
    file_names = [x.replace(extn_list[-1] + file_type,'') for x in files if '.' in x and extn_list[-1] in x]
    # print(file_names)
    for file_name in file_names:			
        ############## Original prediction ##############
        pred_path = path.join(in_path, file_name + file_type)
        pred = np.array(cv2.imread(pred_path, 0))/255.
        sum_pred = np.zeros(pred.shape, dtype=np.float32)
        sum_pred += pred
        ############## 90 rotated prediction ##############
        pred_path = path.join(in_path, file_name + extn_list[1] + file_type)
        pred = np.array(cv2.imread(pred_path, 0))/255.
        pred_90 = np.rot90(pred, 3)
        sum_pred += pred_90
        ############## 180 rotated prediction ##############
        pred_path = path.join(in_path, file_name + extn_list[2] + file_type)
        pred = np.array(cv2.imread(pred_path, 0))/255.
        pred_180 = np.rot90(pred, 2)
        sum_pred += pred_180
        ############## 270 rotated prediction ##############
        pred_path = path.join(in_path, file_name + extn_list[3] + file_type)
        pred = np.array(cv2.imread(pred_path, 0))/255.
        pred_270 = np.rot90(pred, 1)
        sum_pred += pred_270
        ############## vertically flipped prediction ##############
        pred_path = path.join(in_path, file_name + extn_list[4] + file_type)
        pred = np.array(cv2.imread(pred_path, 0))/255.
        V_flip_axis = 0			
        V_flip_pred = cv2.flip(pred, V_flip_axis)
        sum_pred += V_flip_pred
        ############## horizontally flipped prediction ##############
        pred_path = path.join(in_path, file_name + extn_list[5] + file_type)
        pred = np.array(cv2.imread(pred_path, 0))/255.
        H_flip_axis = 1
        H_flip_pred = cv2.flip(pred, H_flip_axis)
        sum_pred += H_flip_pred
        ############## CLAHE prediction ##############
        pred_path = path.join(in_path, file_name + extn_list[6] + file_type)
        pred = np.array(cv2.imread(pred_path, 0))/255.
        sum_pred += pred

        avg_pred = (sum_pred / 7)
        avg_pred_thres = avg_pred.copy()
        avg_pred *= 255
        cv2.imwrite(out_path + file_name + file_type, avg_pred.astype(np.uint8))

def apply_threshold(in_path, out_path, threshold=0.4):
    files = os.listdir(in_path)
    for file in files:
        img = cv2.imread(in_path + file, 0)/255.
        ############### threshold the image ###############
        img[img < threshold] = 0
        img[img >= threshold] = 1
        img *= 255
        cv2.imwrite(out_path + file.replace('_raw_output',''), img.astype(np.uint8))

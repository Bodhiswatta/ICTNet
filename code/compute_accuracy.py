import os
import cv2
import numpy as np
from config import cfg
from os import path

ref_folder = cfg.data_path + '/validation/ground_truth/'
file_names = os.listdir(ref_folder)

def calculate_acc(pred_folder):
	inters_acum = 0
	union_acum = 0
	correct_acum = 0
	total_acum = 0

	result = "city\t\tIoU %\tacc %\n"
	
	for file_name in file_names:
		ref_path = path.join(ref_folder, file_name)
		pred_path = path.join(pred_folder, file_name)

		ref = (np.array(cv2.imread(ref_path, 0))/255.).astype(np.uint8)
		pred = (np.array(cv2.imread(pred_path, 0))/255.).astype(np.uint8)

		inters = ref & pred
		union = ref | pred
		correct = ref == pred

		inters_count = np.count_nonzero(inters)
		union_count = np.count_nonzero(union)
		correct_count = np.count_nonzero(correct)
		total_count = ref.size

		inters_acum+=inters_count
		union_acum+=union_count
		correct_acum+=correct_count
		total_acum+=total_count

		iou = inters_count/float(union_count)
		acc = correct_count/float(total_count)

		if cfg.per_tile_validation_accuracy:
			# print("{0}\t\t{1}%\t{2}%".format(file_name, round(iou*100, 2), round(acc*100, 2)))
			result += "{0}\t\t{1}%\t{2}%\n".format(file_name, round(iou*100, 2), round(acc*100, 2))

	overall_iou = inters_acum/float(union_acum)
	overall_acc = correct_acum/float(total_acum)

	# print("{0}\t\t{1}%\t{2}%".format("Overall", round(overall_iou*100, 2), round(overall_acc*100, 2)))
	result += "{0}\t\t{1}%\t{2}%\n".format("Overall", round(overall_iou*100, 2), round(overall_acc*100, 2))
	return result
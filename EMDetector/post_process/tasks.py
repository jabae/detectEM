import numpy as np
import cv2
from scipy import ndimage
from skimage import measure
import fastremap

from concurrent.futures import ProcessPoolExecutor
from functools import partial


def postprocess(img):

	img_thr = threshold_image(img, 0.35)
	img_con = dilate_folds(img_thr, 5)

	return dilate_folds(filter_folds(img_con, 1500), 15)


def postprocess_chunk(bbox, img):

	chunk_begin = bbox[0]
	chunk_end =bbox[1]

	chunk = img[chunk_begin[0]:chunk_end[0],chunk_begin[1]:chunk_end[1]]

	return postprocess(chunk)
 

def postprocess_dist(img, new_img, bbox_list, n=None):

	with ProcessPoolExecutor(max_workers=n) as pool:
		chunk_list = pool.map(partial(postprocess_chunk, img=img), bbox_list)

	chunk_list = list(chunk_list)
	for i in range(len(bbox_list)):

		chunk_begin = bbox_list[i][0]
		chunk_end = bbox_list[i][1]

		new_img[chunk_begin[0]:chunk_end[0],
				chunk_begin[1]:chunk_end[1]] = chunk_list[i]

	return new_img


def threshold_image(img, thr):

	# Fix threshold
	if np.max(img) > 10 and thr < 1:
		new_thr = 255*thr
	elif np.max(img) < 10 and thr > 1:
		new_thr = thr/255.0
	else:
		new_thr = thr

	return (img>=new_thr).astype('uint8')


def remove_dust(img):

	return cv2.fastNlMeansDenoising(img,None,templateWindowSize=1,searchWindowSize=3)


def dilate_folds(img, w_dilate):

	if np.max(img) > 10:
		img = (img/np.max(img)).astype('uint8')
		
	struct = np.ones((w_dilate,w_dilate), dtype=bool)

	return ndimage.binary_dilation(img, structure=struct).astype(img.dtype)


def filter_folds(img, size_thr):

	img_lab = measure.label(img)

	fold_num, fold_size = np.unique(img_lab, return_counts=True)
	fold_num = fold_num[1:]; fold_size = fold_size[1:]

	img_lab_vec = np.reshape(img_lab, (-1,))
	img_relab = np.reshape(fastremap.remap_from_array_kv(img_lab_vec, fold_num, fold_size), img_lab.shape)

	return (img_relab>=size_thr).astype('uint8')

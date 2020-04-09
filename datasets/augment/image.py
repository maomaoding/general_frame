# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2,os
import random

def flip(img):
	return img[:, :, ::-1].copy()

def transform_preds(coords, center, scale, output_size):
	target_coords = np.zeros(coords.shape)
	trans = get_affine_transform(center, scale, 0, output_size, inv=1)

	target_coords[:, 0:2] = affine_transform(coords[:, 0:2], trans)
	return target_coords


def get_affine_transform(center,
						 scale,
						 rot,
						 output_size,
						 inv=0):
	if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
		scale = np.array([scale, scale], dtype=np.float32)

	scale_tmp = scale
	src_w = scale_tmp[0]
	src_h = scale_tmp[1]
	dst_w = output_size[0]
	dst_h = output_size[1]
	rot_rad = np.pi * rot / 180
	src_dir_w = get_dir([src_w * -0.5,0], rot_rad)
	src_dir_h = get_dir([0, src_h * -0.5], rot_rad)
	dst_dir_w = np.array([ dst_w * -0.5,0], np.float32)
	dst_dir_h = np.array([0, dst_h * -0.5], np.float32)

	src = np.zeros((3, 2), dtype=np.float32)
	dst = np.zeros((3, 2), dtype=np.float32)
	src[0, :] = center
	src[1, :] = center + src_dir_h
	src[2, :] = center - src_dir_w
	dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
	dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir_h
	dst[2, :] =np.array([dst_w * 0.5, dst_h * 0.5], np.float32) - dst_dir_w

	if inv:
		trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
	else:
		trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

	return trans


def affine_transform(pts, t):
	for idx,pt in enumerate(pts):
		pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
		pt = np.dot(t, pt)
		pts[idx]=pt
	return pts[:,:2]


def get_3rd_point(a, b):
	print(a,b)
	direct = a - b
	return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
	sn, cs = np.sin(rot_rad), np.cos(rot_rad)

	src_result = [0, 0]
	src_result[0] = src_point[0] * cs - src_point[1] * sn
	src_result[1] = src_point[0] * sn + src_point[1] * cs

	return src_result


def crop(img, center, scale, output_size, rot=0):
	trans = get_affine_transform(center, scale, rot, output_size)

	dst_img = cv2.warpAffine(img,
							 trans,
							 (int(output_size[0]), int(output_size[1])),
							 flags=cv2.INTER_LINEAR)

	return dst_img

def grayscale(image):
	return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def lighting_(data_rng, image, alphastd, eigval, eigvec):
	alpha = data_rng.normal(scale=alphastd, size=(3,))
	image += np.dot(eigvec, eigval * alpha)


def blend_(alpha, image1, image2):
	image1 *= alpha
	image2 *= (1 - alpha)
	image1 += image2


def saturation_(data_rng, image, gs, gs_mean, var):
	alpha = 1. + data_rng.uniform(low=-var, high=var)
	blend_(alpha, image, gs[:, :, None])


def brightness_(data_rng, image, gs, gs_mean, var):
	alpha = 1. + data_rng.uniform(low=-var, high=var)
	image *= alpha


def contrast_(data_rng, image, gs, gs_mean, var):
	alpha = 1. + data_rng.uniform(low=-var, high=var)
	blend_(alpha, image, gs_mean)


def color_aug(data_rng, image, eig_val, eig_vec):
	functions = [brightness_, contrast_, saturation_]
	random.shuffle(functions)

	gs = grayscale(image)
	gs_mean = gs.mean()
	for f in functions:
		f(data_rng, image, gs, gs_mean, 0.4)
	lighting_(data_rng, image, 0.1, eig_val, eig_vec)

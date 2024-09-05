import sys
import os
import cv2
# from face_alignment import mtcnn
import argparse
from PIL import Image
from tqdm import tqdm
import random
from datetime import datetime
from retinaface.detector import Retinaface
from common import *




detector = Retinaface()
facedata_root = 'test_imgs'
save_path = facedata_root +"_align112"
if not os.path.exists(save_path):
    os.makedirs(save_path)


pad = False
dst_size = [480, 640]

for img_name in os.listdir(facedata_root):
    img_path = os.path.join(facedata_root, img_name)
    cvim = cv2.imread(img_path)
    # if pad:
    #     inpit_img = resize_keep_aspectratio(cvim, dst_size)
    #     cv2.imshow('pad face', inpit_img)
    #     cv2.waitKey(0)
    # else:
    #     inpit_img = cv2.resize(cvim, dst_size)

    bboxes, kpss = detector.inference_one(img_raw = cvim)
    # detector.draw_bbox_kps(cvim, bboxes, kpss)
    if len(bboxes) < 1 or len(kpss) < 1:
        continue
    else:
        dst_kps = xxyy2xyxy(kpss[0])
    crop_face = crop_transform(cvim, dst_kps, image_size = [112, 112])
    cv2.imwrite(os.path.join(save_path, img_name), crop_face)
    # cv2.imshow('face', crop_face)
    # cv2.waitKey(0)
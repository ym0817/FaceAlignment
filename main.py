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
from scrfd.scrfd import Scrfd
from yolo5face.yolo5face import Yolov5face
from common import *


def get_image_list(image_dir, suffix=['jpg', 'jpeg', 'JPG', 'JPEG','png','bmp']):
    '''get all image path ends with suffix'''
    if not os.path.exists(image_dir):
        print("PATH:%s not exists" % image_dir)
        return []
    imglist = []
    for root, sdirs, files in os.walk(image_dir):
        if not files:
            continue
        for filename in files:
            filepath = os.path.join(root, filename)
            if filename.split('.')[-1] in suffix:
                imglist.append(filepath)
    return imglist


def main(face_root, face_align, detector_type):

    if detector_type == 'R':
        detector = Retinaface()
    if detector_type == 'S':
        detector = Scrfd()
    if detector_type == 'Y':
        detector = Yolov5face()

    imgfile_list = get_image_list(face_root)

    for img_path in imgfile_list:
        basename = os.path.basename(img_path)
        id_fold = os.path.split(img_path)[0].split('/')[-1]
        cvim = cv2.imread(img_path)
        bboxes, kpss = detector.inference_one(img_raw=cvim)
        if len(bboxes) < 1 or len(kpss) < 1:
            print('No face detected ')
            continue
        else:
            if mode == 'R':
                # detector.draw_bbox_kps(cvim, bboxes, kpss)
                dst_kps = xxyy2xyxy(kpss[0])
            if mode == 'S':
                dst_kps = kpss[0].reshape(5, 2)
            if mode == 'Y':
                dst_kps = kpss[0].reshape(5, 2)
            crop_face = crop_transform(cvim, dst_kps, image_size=[112, 112])

            save_id_fold = os.path.join(save_path, id_fold)
            if not os.path.exists(save_id_fold):
                os.makedirs(save_id_fold)

            save_imgpath = os.path.join(save_id_fold, basename)
            cv2.imwrite(save_imgpath, crop_face)
        # cv2.imshow('face', crop_face)
        # cv2.waitKey(0)




if __name__ == "__main__":


    facedata_root = 'test_imgs'
    save_path = facedata_root +"_align_112"
    mode = 'Y'  # suport [ 'R', 'Y', 'S' ]

    main(facedata_root, save_path, mode)








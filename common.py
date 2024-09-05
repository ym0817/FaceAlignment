import os
import cv2
import numpy as np
from skimage import transform as trans


def crop_transform(rimg, landmark, image_size):
    """ warpAffine face img by landmark
    """
    assert landmark.shape[0] == 68 or landmark.shape[0] == 5
    assert landmark.shape[1] == 2
    if landmark.shape[0] == 68:  # 68 landmark, select the five-point
        landmark5 = np.zeros((5, 2), dtype=np.float32)
        landmark5[0] = (landmark[36]+landmark[39])/2
        landmark5[1] = (landmark[42]+landmark[45])/2
        landmark5[2] = landmark[30]
        landmark5[3] = landmark[48]
        landmark5[4] = landmark[54]
    else:
        landmark5 = landmark
    tform = trans.SimilarityTransform()
    src = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041]], dtype=np.float32)

    if image_size[1] == 112:
        src[:, 0] += 8.0

    if image_size[1] == 128:
        src[:, 0] += 16.0
        src[:, 1] += 8.0

    if image_size[1] == 160:
        src[:, 0] += 32.0
        src[:, 1] += 24.0

    tform.estimate(landmark5, src)
    M = tform.params[0:2, :]
    img = cv2.warpAffine(rimg, M, (image_size[1], image_size[0]), borderValue=0.0)
    return img


def pad_img(pil_im, target_inputsize):   # im_h, im_w = im.shape[:2]
    canvas = Image.new("RGB", size=target_inputsize, color="#7777")
    target_width, target_height = target_inputsize
    width, height = pil_im.size
    offset_x, offset_y = 0, 0
    if height > width:  # 高 是 长边
        height_ = target_height  # 直接将高调整为目标尺寸
        scale = height_ / height  # 计算高具体调整了多少，得出一个放缩比例
        width_ = int(width * scale)  # 宽以相同的比例放缩
        offset_x = (target_width - width_) // 2  # 计算x方向单侧留白的距离
    else:  # 同上
        width_ = target_width
        scale = width_ / width
        height_ = int(height * scale)
        offset_y = (target_height - height_) // 2
    pil_im = pil_im.resize((width_, height_), Image.BILINEAR)  # 将高和宽放缩
    canvas.paste(pil_im, box=(offset_x, offset_y))
    return canvas




def resize_keep_aspectratio(image_src, dst_size):
    src_h, src_w = image_src.shape[:2]
    dst_h, dst_w = dst_size
    h = dst_w * (float(src_h) / src_w)  # 按照ｗ做等比缩放
    w = dst_h * (float(src_w) / src_h)  # 按照h做等比缩放
    if int(h) <= dst_h:
        image_dst = cv2.resize(image_src, (dst_w, int(h)))
    else:
        image_dst = cv2.resize(image_src, (int(w), dst_h))
    h_, w_ = image_dst.shape[:2]
    top = int((dst_h - h_) / 2)
    down = int((dst_h - h_ + 1) / 2)
    left = int((dst_w - w_) / 2)
    right = int((dst_w - w_ + 1) / 2)
    value = [0, 0, 0]
    borderType = cv2.BORDER_CONSTANT
    image_dst = cv2.copyMakeBorder(image_dst, top, down, left, right, borderType, None, value)
    return image_dst



def xxyy2xyxy(src_kps):

    dst_kps = np.array([
        [src_kps[0], src_kps[5]],
        [src_kps[1], src_kps[6]],
        [src_kps[2], src_kps[7]],
        [src_kps[3], src_kps[8]],
        [src_kps[4], src_kps[9]]], dtype=np.float32)

    return dst_kps






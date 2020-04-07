# -*- coding: UTF-8 -*-

"""
# version: python3.6
# author: TechDing
# email: dingustb@126.com
# file: extract_IDImg.py
# time: 2019/9/10 9:19
# doc: 
"""

import os, cv2
import numpy as np
def get_source_p(img0):
    def find_loc(arr):
        result = len(arr) - 1
        for idx, num in enumerate(arr):
            if num > 150:
                result = idx
                break
        return result

    def get_p(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
                    (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
                    (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        return int(px), int(py)

    H, W = img0.shape
    quart_H = H // 4
    quart_W = W // 4
    up_p1 = (quart_W, find_loc(img0[:quart_H, quart_W]))  # OK
    up_p2 = (W - quart_W, find_loc(img0[:quart_H, -quart_W]))  # OK
    left_p1 = (find_loc(img0[quart_H, :quart_W]), quart_H)  # OK
    left_p2 = (find_loc(img0[-quart_H, :quart_W]), H - quart_H)  # OK
    left_up1, left_up2 = get_p(up_p1, up_p2, left_p1, left_p2)
    bot_p1 = (quart_W, H - find_loc(img0[-quart_H:, quart_W][::-1]))  # OK
    bot_p2 = (W - quart_W, H - find_loc(img0[-quart_H:, -quart_W][::-1]))  # OK
    left_bot1, left_bot2 = get_p(left_p1, left_p2, bot_p1, bot_p2)
    right_p1 = (W - find_loc(img0[quart_H, -quart_W:][::-1]), quart_H)
    right_p2 = (W - find_loc(img0[-quart_H, -quart_W:][::-1]), H - quart_H)
    right_up1, right_up2 = get_p(right_p1, right_p2, up_p1, up_p2)
    return np.float32([[left_bot1, left_bot2], [left_up1, left_up2], [right_up1, right_up2]])

def extract_IDImg(img_path): # OK
    '''
    从img_path对应的图片路径中提取出身份证ID所在的区域ROI,放置到result这个list中，所以这个list有两张ROI,分别为正面和反面。
    :param img_path:
    :return:
    '''
    def getAffine(img_arr, src_points, dst_points=None):
        if dst_points is None:
            dst_points = np.float32([[0, 282], [0, 0], [448, 0]])
        w, h = np.max(dst_points, axis=0)
        affineMatrix = cv2.getAffineTransform(src_points, dst_points)
        return cv2.warpAffine(img_arr, affineMatrix, (w, h))

    def split_line(card_binary):  # OK
        H, W = card_binary.shape
        start_H = int(H * 0.43)
        roi = card_binary[start_H:-start_H]
        return np.argmin(np.sum(roi, axis=1)) + start_H

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bk = np.unique(gray[-50:-20, -50:-20])[0]
    gray[gray != bk] = 255
    gray[gray==bk]=0
    # _, binary = cv2.threshold(gray, 255, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    box_list = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        short = min(width, height)
        long = max(width, height)
        if short < min(gray.shape[:2]) * 0.2 or long < max(gray.shape[:2]) * 0.2 or short >= min(
                gray.shape[:2]) - 10 or long >= max(gray.shape[:2]) - 10:
            continue
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        box_list.append([rect[2], box])
    result=[]
    if len(box_list) == 2:
        for idx, (rect, box) in enumerate(box_list):
            src_points = np.float32(box[1:]) if rect < -45 else np.float32(box[:3])
            card_img = getAffine(img, src_points)
            result.append(card_img)
    elif len(box_list) == 1:
        # print('found 1 box: ',img_path)
        rect, box = box_list[0]
        xmin, ymin = np.min(np.array(box), axis=0)
        xmax, ymax = np.max(np.array(box), axis=0)
        dst_points2 = np.float32([[0, ymax - ymin], [0, 0], [xmax - xmin, 0]])
        src_points = np.float32(box[1:]) if rect < -45 else np.float32(box[:3])
        # card_binary = getAffine(binary, src_points, dst_points2)
        card_binary = getAffine(gray, src_points, dst_points2)
        card = getAffine(img, src_points, dst_points2)
        line = split_line(card_binary)
        up = getAffine(card[:line], get_source_p(card_binary[:line]))
        down = getAffine(card[line:], get_source_p(card_binary[line:]))
        result.append(up)
        result.append(down)
    else:
        print('error: found {} box in {}'.format(len(box_list), img_path))
    return result

def get_IDImg(img_path,model,save_folder=None): # OK
    '''
    从img_path中提取出身份证的正反面所在区域，并用model来预测是正立还是倒立，如果是倒立则旋转到正立状态。
    :param img_path: 原始的图片路径
    :param model: 图像分类模型，预测图像是正面，反面，还是正立，倒立，
        输出4类：0-正面正立，1-正面倒立，2-反面正立，3-反面倒立
    :param save_folder: 如果要保存提取的正立的身份证ROI，可以设置本保存文件路径，默认None，不保存
    :return: 提取出来的一张图片的正面和反面的正立图像
    '''
    if save_folder: os.makedirs(save_folder,exist_ok=True)
    IDImgs=extract_IDImg(img_path) # 提取出的身份证正面和反面的ROI
    img_name = os.path.splitext(os.path.split(img_path)[1])[0]
    result=[]
    for idx,img in enumerate(IDImgs):
        label = np.argmax(model.predict(np.array([cv2.cvtColor(img[29:253, 112:336], cv2.COLOR_BGR2GRAY) \
                                             [:, :, np.newaxis].astype(np.float32) / 255.0])), axis=1)[0]
        temp=img if label in (0, 2) else cv2.rotate(img, 1) # 如果是反面则要rotate
        result.append(temp)
        if save_folder:
            cv2.imwrite(os.path.join(save_folder,img_name+'_'+str(idx)+'.jpg'),temp)
    return result

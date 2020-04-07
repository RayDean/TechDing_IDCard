# -*- coding: UTF-8 -*-
from __future__ import division
from __future__ import print_function

"""
# version: python3.6
# author: TechDing
# email: dingustb@126.com
# file: pred_box.py
# time: 2019/8/28 14:02
# doc: 
"""
import cv2
def preprocess_input(image, net_h, net_w):
    new_h, new_w, _ = image.shape
    if (float(net_w)/new_w) < (float(net_h)/new_h):
        new_h = (new_h * net_w)//new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h)//new_h
        new_h = net_h
    resized = cv2.resize(image[:,:,::-1]/255., (new_w, new_h))
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[(net_h-new_h)//2:(net_h+new_h)//2, (net_w-new_w)//2:(net_w+new_w)//2, :] = resized
    new_image = np.expand_dims(new_image, 0)
    return new_image

from scipy.special import expit
def _sigmoid(x):
    return expit(x)

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.c = c
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        return self.score
    def __repr__(self):
        return 'bbox: [{},{},{},{}],classes: {}, c: {},label: {},score: {}'\
            .format(self.xmin,self.ymin,self.xmax,self.ymax,self.classes,self.c,self.label,self.score)

def _softmax(x, axis=-1):
    x = x - np.amax(x, axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / e_x.sum(axis, keepdims=True)

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5
    boxes = []
    netout[..., :2] = _sigmoid(netout[..., :2])
    netout[..., 4] = _sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h * grid_w):
        row = i // grid_w
        col = i % grid_w
        for b in range(nb_box):
            objectness = netout[row, col, b, 4]
            if (objectness <= obj_thresh): continue
            x, y, w, h = netout[row, col, b, :4]
            x = (col + x) / grid_w
            y = (row + y) / grid_h
            w = anchors[2 * b + 0] * np.exp(w) / net_w
            h = anchors[2 * b + 1] * np.exp(h) / net_h
            classes = netout[row, col, b, 5:]
            box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)
            boxes.append(box)
    return boxes

def correct_decode_boxes(boxes, image_h, image_w, net_h, net_w):
    if (float(net_w) / image_w) < (float(net_h) / image_h):
        new_w = net_w
        new_h = (image_h * net_w) / image_w
    else:
        new_h = net_w
        new_w = (image_w * net_h) / image_h

    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
        y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h

        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3

def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin
    union = w1 * h1 + w2 * h2 - intersect
    return float(intersect) / union if union != 0 else 0

def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0

import numpy as np
def pred_batch_boxes(model, images, anchors, obj_thresh, nms_thresh):
    net_h, net_w = 416, 416
    image_h, image_w, _ = images[0].shape
    nb_images = len(images)
    batch_input = np.zeros((nb_images, net_h, net_w, 3))

    for i in range(nb_images):
        batch_input[i] = preprocess_input(images[i], net_h, net_w)

    batch_output = model.predict_on_batch(batch_input)
    batch_boxes = [None] * nb_images

    for i in range(nb_images):
        yolos = [batch_output[0][i], batch_output[1][i], batch_output[2][i]]
        boxes = []
        for j in range(len(yolos)):
            yolo_anchors = anchors[(2 - j) * 6:(3 - j) * 6]
            boxes += decode_netout(yolos[j], yolo_anchors, obj_thresh, net_h, net_w)
        correct_decode_boxes(boxes, image_h, image_w, net_h, net_w)
        do_nms(boxes, nms_thresh)
        batch_boxes[i] = boxes
    return batch_boxes

def correct_bbox(box,H,W):
    box.xmin=np.clip(box.xmin,0,W-1)
    box.ymin=np.clip(box.ymin,0,H-1)
    box.xmax=np.clip(box.xmax,0,W-1)
    box.ymax=np.clip(box.ymax,0,H-1)
    return box

def resize_box(boxes,ratio,old_H,img,label):
    result=[]
    if label in (3,6):
        box=boxes[0]
        img_temp = cv2.resize(img[box.ymin:box.ymax,box.xmin:box.xmax], (60, 60), interpolation=cv2.INTER_CUBIC)
        img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
        result.append(img_temp.astype(np.float32) / 255.0)
    else:
        for box in boxes:
            box_h = box.ymax - box.ymin
            box_w = box.xmax - box.xmin
            new_w = old_H * box_w / box_h
            center_h = (box.ymin + box.ymax) // 2
            center_w = (box.xmin + box.xmax) // 2
            x1 = center_w - new_w // 2
            y1 = center_h - old_H //2
            x2 = x1 + new_w
            y2 = center_h + old_H //2
            imgW = max(32, int(np.floor(ratio * 32)))
            img_temp = cv2.resize(img[y1:y2, x1:x2], (imgW, 32), interpolation=cv2.INTER_CUBIC)
            img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
            result.append(img_temp.astype(np.float32) / 255.0)
    return np.array(result)

def _change_box(label,img):
    def resize_box2(box, ratio, new_H, img):
        result = []
        imgW = max(new_H, int(np.floor(ratio * new_H))) if ratio > 0 else new_H
        inter = cv2.INTER_CUBIC if ratio > 0 else cv2.INTER_AREA
        img_temp = cv2.resize(img, (imgW, new_H), interpolation=inter)
        img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
        result.append(img_temp)
        return np.array(result)[:, :, :, np.newaxis].astype(np.float32) / 255.0

    result=None
    if label in (0,7): # unit address
        result=resize_box2(None,10.5,32,img)
    elif label in (1,5):
        result = resize_box2(None, 9.5, 32, img)
    elif label==2:
        result=resize_box2(None, 12.3, 32, img)
    elif label==3:
        result = resize_box2(None, 0, 80, img)
    elif label==6:
        result=resize_box2(None, 0, 32, img)
    elif label ==4:
        result = resize_box2(None, 3, 32, img)
    return result

def change_box(roi_dict,img):
    def resize_box2(boxes, ratio, new_H, img):
        result = []
        for box in boxes:
            imgW = max(new_H, int(np.floor(ratio * new_H))) if ratio > 0 else new_H
            inter = cv2.INTER_CUBIC if ratio > 0 else cv2.INTER_AREA
            img_temp = cv2.resize(img[box.ymin:box.ymax, box.xmin:box.xmax], (imgW, new_H), interpolation=inter)
            img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
            result.append(img_temp)
        return np.array(result)[:, :, :, np.newaxis].astype(np.float32) / 255.0

    result = {}
    for label,boxes in roi_dict.items():
        if label in (0, 7):  # unit address
            result[label] = resize_box2(boxes, 10.5, 32, img)
        elif label in (1, 5):
            result[label] = resize_box2(boxes, 9.5, 32, img)
        elif label == 2:
            result[label] = resize_box2(boxes, 12.3, 32, img)
        elif label == 3:
            result[label] = resize_box2(boxes, 0, 80, img)
        elif label == 6:
            result[label] = resize_box2(boxes, 0, 32, img)
        elif label == 4:
            result[label] = resize_box2(boxes, 3, 32, img)
    return result

def finetue_box(label,boxes):
    def id(boxes): # 结果更好
        box=boxes[0]
        center = (box.xmax + box.xmin) // 2
        box.ymin += 1
        box.ymax = box.ymin + 22 # 254 22
        half_w = 127
        box.xmin = center - half_w
        box.xmax = center + half_w
        return boxes
    # def name(boxes): # 效果不佳
    #     box = boxes[0]
    #     w = box.xmax - box.xmin
    #     center = (box.xmax + box.xmin) // 2
    #     h = box.ymax - box.ymin
    #     long_type = w / h >= 2.5  # 这种模式对w/h很敏感，如果出错，那么很容易出问题
    #     box.ymin += 2
    #     box.ymax = box.ymin + 20
    #     half_w = 30 if long_type else 20
    #     box.xmin = center - half_w
    #     box.xmax = center + half_w
    #     return boxes
    def name(boxes):
        return boxes
    def minzu(boxes): # 显著提高
        box = boxes[0]
        w = box.xmax - box.xmin
        center = (box.xmax + box.xmin) // 2+2 # center往右移动2个px
        h = box.ymax - box.ymin
        ratio=w / h
        if ratio>=3.5:
            half_w=40
        elif ratio>=2.5:
            half_w=30
        elif ratio>=1.5:
            half_w=20
        else:
            half_w=10
        box.ymin += 3
        box.ymax = box.ymin + 20
        box.xmin = center - half_w
        box.xmax = center + half_w
        return boxes
    def sex(boxes): # 结果一样
        box = boxes[0]
        box.ymin += 3
        box.ymax = box.ymin + 20
        box.xmin+=2
        box.xmax=box.xmin+20
        return boxes
    def birth(boxes): # 结果更好
        box = boxes[0]
        center = (box.xmax + box.xmin) // 2
        box.ymin += 2
        box.ymax = box.ymin + 20
        half_w = 77
        box.xmin = center - half_w
        box.xmax = center + half_w
        return boxes
    def range_(boxes): # 结果更好
        box = boxes[0]
        w = box.xmax - box.xmin
        center = (box.xmax + box.xmin) // 2
        h = box.ymax - box.ymin
        long_type = w / h >= 8.1  # 这种模式对w/h很敏感，如果出错，那么很容易出问题
        # box.ymin += 5
        box.ymin += 3 # 此处有修改
        box.ymax = box.ymin + 20
        half_w = 95 if long_type else 67
        box.xmin = center - half_w
        box.xmax = center + half_w
        return boxes
    def address(boxes):
        lens=len(boxes)
        for idx,box in enumerate(boxes):
            box.ymin+=4
            box.ymax=box.ymin+20
            if idx!=lens-1:
                center = (box.xmax + box.xmin) // 2
                half_w = 105
                box.xmin = center - half_w
                box.xmax = center + half_w
        return boxes
    func_dict={0:address,1:birth,2:id,3:minzu,4:name,5:range_,6:sex,7:address}
    return func_dict[label](boxes)

anchors=[18,28,37,28,58,29,114,29,140,28,147,31,147,34,197,27,197,31] # IDCard_OCR_V190829.h5
# anchors=[18,28,37,28,60,29,136,27, 140, 30,  141, 35, 176, 27, 194, 33, 218, 28] # IDCard_yolo3_2_V190921.h5
# 不同的yolo模型要用不同anchors
obj_thresh=0.70
nms_thresh=0.45

import uuid,os
def pred_roi(model,images,save_folder=None): # OK
    if isinstance(images,str):
        name = os.path.splitext(os.path.split(images)[1])[0]
        # print('name: ',name)
        images=[cv2.imread(images)]
    else:
        name=uuid.uuid4().hex[:20].upper()
    if save_folder: os.makedirs(save_folder,exist_ok=True)
    roi_boxes=pred_batch_boxes(model,images,anchors,obj_thresh,nms_thresh)
    result={} # 一个样本的roi imgs 组成的dict,key为label, value为[box1, box2...]
    default_box0={0:BoundBox(85,142,295,162),
                  1:BoundBox(86,110,240,130),
                  2:BoundBox(130,227,384,249),
                  # 3:BoundBox(190,78,230,98),
                  3:BoundBox(190,78,210,98), # 20,20,minzu
                  4:BoundBox(88,45,148,65),
                  6:BoundBox(88,78,108,98)} # 正面图片的default
    default_box1={5:BoundBox(172,230,362,250),7:BoundBox(174,196,384,216)} # 反面图片的default box
    for idx,roi_box in enumerate(roi_boxes):
        temp={} # key: label, value: [box1, box2...]
        [temp.setdefault(np.argmax(box.classes),[]).append(box) for box in roi_box]
        for label,box_list in temp.items():
            if label in (0,7): # address, unit
                temp_list=[box for box in box_list if max(box.classes)>0.9]
                temp_list.sort(key=lambda x: max(x.classes), reverse=True)  # get max classes
                temp_list=temp_list[:3] # 最多只有3个
                temp_list.sort(key=lambda x: x.ymin) # 排序
                temp[label]=temp_list
            else:
                box_list.sort(key=lambda x: max(x.classes),reverse=True) # get max classes
                temp[label]=[box_list[0]]
            temp[label]=finetue_box(label,temp[label]) # fine tune box
        # 有的label没有box，此时使用默认的box来代替
        default_box=default_box1
        if 1 in temp.keys() or 2 in temp.keys(): # 有birth, 是正面
            default_box=default_box0
        for key,value in default_box.items():
            if key not in temp.keys():
                print('key ',key,'not in ',name)
                temp[key]=[value]
        if 7 in temp.keys() and 0 in temp.keys():
            temp.pop(0) # 删除key==0
        if save_folder:
            for label,boxes in temp.items():
                for i,box in enumerate(boxes):
                    cv2.imwrite(os.path.join(save_folder,name+'_'+str(label)+'_'+str(i)+'.jpg'),
                                images[idx][box.ymin:box.ymax,box.xmin:box.xmax])
        # result.update(change_box(temp,images[idx]))
        result.update(temp)
    return result

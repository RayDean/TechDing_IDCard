# -*- coding: UTF-8 -*-

"""
# version: python3.6
# author: TechDing
# email: dingustb@126.com
# file: pred_content.py
# time: 2019/9/10 11:20
# doc: 
"""

from keras.models import load_model
from .Words0 import words0
from .Words4 import words4
from .extract_IDImg import get_IDImg
from .pred_box import pred_roi
from collections import OrderedDict
import numpy as np
import cv2
class TechDing_Model:

    def __init__(self):
        print('start to load model, this will take some time ...')
        self.extract_model=load_model('./Models/IDCard_UpDn_V191108_2')
        # 图像分类模型，预测图像是正面，反面，还是正立，倒立，输出4类：0-正面正立，1-正面倒立，2-反面正立，3-反面倒立
        self.box_model=load_model('./Models/IDCard_OCR_V190829.h5')
        self.models={
            0: load_model('./Models/Model0'), # address
            1: load_model('./Models/Model1'), # birth
            2: load_model('./Models/IDCard_ID_50W2_V190927_base'), # id
            3: load_model('./Models/IDCard_Minzu_50W_32_V190926_2'), # minzu
            4: load_model('./Models/IDCard_Name4_100W_V191030_1_base'), # name
            5: load_model('./Models/IDCard_Range_50W2_V190929_base'), # range
            6: load_model('./Models/Model3') # sex
            }
        print('TechDing model is loaded...')
        self.words = {
            0: words0,
            1: '0123456789年月日-.长期',
            2: '0123456789X',
            3: np.array(['东乡', '乌孜别克', '京', '仡佬', '仫佬', '佤', '侗', '俄罗斯', '保安', '傈僳', '傣', '哈尼',
                         '哈萨克', '回', '土', '土家', '基诺', '塔吉克', '塔塔尔', '壮', '崩龙', '布依', '布朗', '彝',
                         '怒', '拉祜', '撒拉', '普米', '景颇', '朝鲜', '柯尔克孜', '毛难', '水', '汉', '满', '独龙',
                         '珞巴', '瑶', '畲', '白', '纳西', '维吾尔', '羌', '苗', '蒙古', '藏', '裕固', '赫哲', '达斡尔',
                         '鄂伦春', '鄂温克', '锡伯', '门巴', '阿昌', '高山', '黎']),
            4: words4,
            5: '0123456789-.长期',
            6: np.array(['男', '女']),
        }
        self.orders=OrderedDict({4: 4, 3: 3, 6: 6, 1: 1, 0: 0, 2: 2, 7: 0, 5: 5}) # key1, label, value: model type

    def pred_img(self,img_path):
        print('img_path: ',img_path)
        imgs=get_IDImg(img_path,self.extract_model)
        rois=pred_roi(self.box_model,imgs,save_folder='E:\Test3_roi')
        # rois: 一个样本的roi imgs 组成的dict,key为label, value为[box1, box2...]
        print('rois len: ',len(rois))
        assert len(rois)>0
        result=[] # 这个里面的result之间有内在关系，比如：生日必定是ID中，Address和Unit有一定关系，
        # Range内部有关系，Range和Birth也有部分关系，充分利用这些关系也可以提高准确率
        for key1,key2 in self.orders.items():
            pred_str=''
            # try:
            roi=self._change_box(key1,rois[key1])
            pred_str=self._decode_pred(self.models[key2].predict(roi),self.words[key2],key1)
            # except Exception as e:
            #     print('except: ',img_path,e)
            if key1 == 1:  # Birth
                pred_str = pred_str.replace('年', ',').replace('月', ',').replace('日', '') \
                    .replace(',,', ',').replace('-', ',').replace('.', ',')
                lens = len(pred_str.split(','))
                if lens == 2:  # 只有一个，需要添加一个
                    pred_str = pred_str[:4] + ',' + pred_str[4:]
                elif lens > 3:  # 有多个逗号，只取最前面的3部分
                    pred_str = ','.join(pred_str.split(',')[:3])
            else:  # 其他的类别中不能有逗号
                pred_str = pred_str.replace(',', '')
            result.append(pred_str)
        return ','.join(result)

    def _change_box(self, label, img):
        def resize_box2(ratio, new_H,img):
            result = []
            imgW = max(new_H, int(np.floor(ratio * new_H))) if ratio > 0 else new_H
            inter = cv2.INTER_CUBIC if ratio > 0 else cv2.INTER_AREA
            img_temp = cv2.resize(img, (imgW, new_H), interpolation=inter)
            img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
            result.append(img_temp)
            return np.array(result)[:, :, :, np.newaxis].astype(np.float32) / 255.0

        # 粗略的调节对比度和亮度
        def contrast_brightness_image(src1, a, g):
            h, w, ch = src1.shape  # 获取shape的数值，height和width、通道
            # 新建全零图片数组src2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
            src2 = np.zeros([h, w, ch], src1.dtype)
            dst = cv2.addWeighted(src1, a, src2, 1 - a, g)  # addWeighted函数说明如下
            return dst

        print('label: ',label,'img shape: ',img.shape)
        param={0:10.8,1:9.5,2:11.545,4:3,5:9.5,6:0}
        if label==3: # minzu
            if img.shape[1] <= 40:  # acc 0.9892
                img = img[:, :40]  # 这一次只截取W前面的40个px
            else:
                img = img[:, 15:55]  # 这一次只截取W前面的40个px
            # img=contrast_brightness_image(img,1.5,3) # 这种方式acc 最大为0.9892
            result = resize_box2(0, 32, img)
        else:
            # if label in (0,7): # unit address
            #     # print('label: ',label,'ratio: ', img.shape[1]/img.shape[0])
            #     label=-1 if img.shape[1]/img.shape[0]<=6 else 0
            #     # print('after: ',label,'after ratio: ',param[label])
            result=resize_box2(param[label],32,img)
        return result

    def pred_roi(self,roi_path):
        roi=cv2.imread(roi_path)
        key1=int(os.path.split(roi_path)[1].split('_')[2])
        # 0-Address, 1-birth, 2-ID, 3-minzu,4-name,5-range,6-sex, 7-unit
        key2=self.orders[key1]
        roi = self._change_box(key1,roi)
        pred_str=''
        try:
            # if key1 in (0,7):
            #     model=self.models[0][0] if roi.shape[1]/roi.shape[0]<=6 else self.models[0][1]
            # else:
            model=self.models[key2]
            pred_str = self._decode_pred(model.predict(roi), self.words[key2], key1)
        except Exception as e:
            print('except: ', roi_path, e)
        if key1 == 1:  # Birth
            pred_str = pred_str.replace('年', ',').replace('月', ',').replace('日', '') \
                .replace(',,', ',').replace('-', ',').replace('.', ',')
            lens = len(pred_str.split(','))
            if lens == 2:  # 只有一个，需要添加一个
                pred_str = pred_str[:4] + ',' + pred_str[4:]
            elif lens > 3:  # 有多个逗号，只取最前面的3部分
                pred_str = ','.join(pred_str.split(',')[:3])
        else:  # 其他的类别中不能有逗号
            pred_str = pred_str.replace(',', '')
        return pred_str

    def _decode_pred(self,predy,chars,key):
        if key in (3,6):
            return chars[np.argmax(predy, axis=1)][0]
        else:
            bs_chars=[]
            blank_id = len(chars)
            predy_max = predy.argmax(axis=2)
            for predmax in predy_max:
                char_list=[]
                former=-1
                for idx in predmax:
                    if idx>=blank_id:
                        former = -1
                    elif idx!=former:
                        former=idx
                        char_list.append(chars[idx])
                bs_chars.append(''.join(char_list).strip())
            return ''.join(bs_chars)

class TechDing_One_Model:

    def __init__(self,model_type):
        print('start to load model, this will take some time ...')
        self.orders = OrderedDict({4: 4, 3: 3, 6: 6, 1: 1, 0: 0, 2: 2, 7: 0, 5: 5})
                                                        # Train_Card_Rois_V190829_4  Train_Card_Rois_V190921_2
        # self.model_dict={'name':'IDCard_Name_500W2_V190928_base', # acc 0.8838 0.8413
        # self.model_dict={'name':'IDCard_Name_500W2_V191028_base', # 这个的acc 0.9377 0.9113
        # self.model_dict={'name':'IDCard_Name_500Wval2_V191029_base', # 这个的acc 0.8624 0.8249
        # self.model_dict={'name':'IDCard_Name4_100W_V191029_1_base', # 这个的acc 0.9289 0.9149
        # self.model_dict={'name':'IDCard_Name4_100W_V191029_2_base', # 这个的acc 0.9216 0.9039
        # self.model_dict={'name':'IDCard_Name4_100W_V191029_3_base', # 这个的acc 0.9152 0.9011
        self.model_dict={'name':'IDCard_Name4_100W_V191030_1_base', # 这个的acc 0.938 0.924
                         'minzu':'IDCard_Minzu_50W_32_V190926_2',
                         'sex':'Model3',
                         # 'birth':'IDCard_Birth_50W2_V190928_base',
                         'birth':'Model1', # 貌似这个更好一些
                         'address':'Model0',
                         # 'id':'Model2',
                         'id':'IDCard_ID_50W2_V190927_base',
                         'unit':'Model0',
                         'range':'IDCard_Range_50W2_V190929_base'}
        self.model=load_model('./Models/'+self.model_dict[model_type.lower()])
        print('TechDing One model is loaded...')
        self.words={
            0: words0,
            # 1: '0123456789年月日',
            1: '0123456789年月日-.长期',
            2: '0123456789X',
            3: np.array(['东乡', '乌孜别克', '京', '仡佬', '仫佬', '佤', '侗', '俄罗斯', '保安', '傈僳', '傣', '哈尼',
                         '哈萨克', '回', '土', '土家', '基诺', '塔吉克', '塔塔尔', '壮', '崩龙', '布依', '布朗', '彝',
                         '怒', '拉祜', '撒拉', '普米', '景颇', '朝鲜', '柯尔克孜', '毛难', '水', '汉', '满', '独龙',
                         '珞巴', '瑶', '畲', '白', '纳西', '维吾尔', '羌', '苗', '蒙古', '藏', '裕固', '赫哲', '达斡尔',
                         '鄂伦春', '鄂温克', '锡伯', '门巴', '阿昌', '高山', '黎']),
            4: words4,
            5: '0123456789-.长期',
            6: np.array(['男', '女']),
        }

    def _change_box(self,label,img):
        def resize_box2(box, ratio, new_H, img):
            result = []
            imgW = max(new_H, int(np.floor(ratio * new_H))) if ratio > 0 else new_H
            inter = cv2.INTER_CUBIC if ratio > 0 else cv2.INTER_AREA
            img_temp = cv2.resize(img, (imgW, new_H), interpolation=inter)
            img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
            # img_temp = cv2.equalizeHist(img_temp) # Minzu 使用直方图均衡化之后效果变差
            result.append(img_temp)
            return np.array(result)[:, :, :, np.newaxis].astype(np.float32) / 255.0

        # 粗略的调节对比度和亮度
        def contrast_brightness_image(src1, a, g):
            h, w, ch = src1.shape  # 获取shape的数值，height和width、通道
            # 新建全零图片数组src2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
            src2 = np.zeros([h, w, ch], src1.dtype)
            dst = cv2.addWeighted(src1, a, src2, 1 - a, g)  # addWeighted函数说明如下
            return dst

        result=None
        if label in (0,7): # unit address
            result=resize_box2(None,10.5,32,img)
        elif label==1: # birth
            result = resize_box2(None, 9.5, 32, img) # 7.7
            # result = resize_box2(None, 7.7, 32, img) # 7.7
        elif label==2: # id
            result=resize_box2(None, 11.545, 32, img)
        elif label==3: # minzu
            if img.shape[1]<=40: # acc 0.9892
                img = img[:,:40]# 这一次只截取W前面的40个px
            else:
                img = img[:,15:55] # 这一次只截取W前面的40个px
            # img=contrast_brightness_image(img,1.5,3) # 这种方式acc 最大为0.9892
            result = resize_box2(None, 0, 32, img)
        elif label==4: # name
            # img = contrast_brightness_image(img, 1.5, 3)  # 这种方式acc 最大为0.9892
            result = resize_box2(None, 3, 32, img)
        elif label==5: # range
            result = resize_box2(None, 9.5, 32, img)
        elif label==6: # sex
            result=resize_box2(None, 0, 32, img)
        return result

    def pred_roi(self,roi_path):
        roi=cv2.imread(roi_path)
        key1=int(os.path.split(roi_path)[1].split('_')[2])
        key2=self.orders[key1] # 和models对应的
        roi = self._change_box(key1,roi)
        pred_str=''
        try:
            pred_str = self._decode_pred(self.model.predict(roi), self.words[key2], key1)
        except Exception as e:
            print('except: ', roi_path, e)
        if key1 == 1: # Birth
            pred_str = pred_str.replace('年', ',').replace('月', ',').replace('日','')\
                .replace(',,',',').replace('-',',').replace('.',',')
            lens=len(pred_str.split(','))
            if lens==2: # 只有一个，需要添加一个
                pred_str=pred_str[:4]+','+pred_str[4:]
            elif lens>3: # 有多个逗号，只取最前面的3部分
                pred_str=','.join(pred_str.split(',')[:3])
        elif key1==4: # name
            pred_str=pred_str.replace(',','')
        return pred_str

    def _decode_pred(self,predy,chars,key):
        if key in (3,6): # minzu, sex
            return chars[np.argmax(predy, axis=1)][0]
        else:
            bs_chars=[]
            blank_id = len(chars)
            predy_max = predy.argmax(axis=2)
            for predmax in predy_max:
                char_list=[]
                former=-1
                for idx in predmax:
                    if idx>=blank_id:
                        former = -1
                    elif idx!=former:
                        former=idx
                        char_list.append(chars[idx])
                bs_chars.append(''.join(char_list).strip())
            return ''.join(bs_chars)

import os
def pred_folders_img(imgs_folder,result_save_path): # OK
    '''
    预测imgs_folder中所有原始图片，并将预测结果保存到result_save_path中
    :param imgs_folder: 所有原始图片位于本folder中
    :param result_save_path: 最终的预测结果保存到这个csv中
    :return:
    '''
    assert os.path.exists(imgs_folder) and os.path.isdir(imgs_folder)
    os.makedirs(os.path.dirname(result_save_path),exist_ok=True)
    model=TechDing_Model() # 这是最主要的预测模型
    save_file=open(result_save_path,'w',encoding='utf-8')
    all_imgs=os.listdir(imgs_folder)
    for idx, img_name in enumerate(all_imgs): # 每张图片逐一提取重要信息
        result=model.pred_img(os.path.join(imgs_folder,img_name))
        save_file.write(os.path.splitext(img_name)[0]+','+result+'\n')
        print('\r{}/{} finished...'.format(idx+1,len(all_imgs)),end=' ')
    save_file.flush()
    save_file.close()
    print('\nGOOD, All finished...')

# from glob import glob
# from collections import OrderedDict
# def pred_folders_roi(roi_folder,result_save_path):
#     cls_dict = OrderedDict({'name': 4, 'minzu': 3, 'sex': 6, 'birth': 1,
#                 'address': 0, 'id': 2, 'unit': 7, 'range': 5})
#     cls=list(cls_dict.keys())
#     all_imgs=np.unique([item.split('_')[0] for item in os.listdir(roi_folder)])
#     model = TechDing_Model()
#     save_file = open(result_save_path, 'w', encoding='utf-8')
#     save_file.write(','.join(['img_name']+cls)+'\n')
#     for idx,img_name in enumerate(all_imgs):
#         # if idx<4643: continue
#         img_result=[]
#         for i in cls:
#             label=cls_dict[i]
#             rois = glob(roi_folder + '/' + img_name + '_?_'+str(label)+'_?.jpg')
#             # print('rois: ',rois)
#             cls_result=''.join([model.pred_roi(roi) for roi in rois])
#             img_result.append(cls_result)
#         save_file.write(img_name+ ',' + ','.join(img_result) + '\n')
#         print('\r{}/{} finished...'.format(idx + 1, len(all_imgs)), end=' ')
#     save_file.flush()
#     save_file.close()
#     print('\nGOOD, All finished...')

from glob import glob
from collections import OrderedDict
import sys
def pred_folders_roi(roi_folder,result_save_path):
    cls_dict = OrderedDict({'name': 4, 'minzu': 3, 'sex': 6, 'birth': 1,
                'address': 0, 'id': 2, 'unit': 7, 'range': 5})
    # cls=list(cls_dict.keys())
    all_names=set(os.listdir(roi_folder))
    all_imgs=np.unique([item.split('_')[0] for item in all_names])
    # 将所有的imgs整理好放在一个大的dict里面，key为img_name, value为一个dict，key为0-7，value为对应的img_path
    all_dict={}
    for img_name in all_imgs:
        value_dict={}
        for labels in cls_dict.values():
            temp_li=[]
            for i in [0,1]:
                for j in range(5):
                    temp_name=img_name+'_'+str(i)+'_'+str(labels)+'_'+str(j)+'.jpg'
                    if temp_name in all_names:
                        temp_li.append(temp_name)
            value_dict[labels]=temp_li
        all_dict[img_name]=value_dict
    print('all_img_names prepared...')
    ##### 测试一下看看是否正常：# 测试OK
    # num=0
    # for img_name,value_dict in all_dict.items():
    #     print(img_name,value_dict)
    #     num+=1
    #     if num>10: break
    # sys.exit(0)

    model = TechDing_Model()
    save_file = open(result_save_path, 'w', encoding='utf-8')
    # save_file.write(','.join(['img_name']+cls)+'\n')
    for idx,(img_name,label_dict) in enumerate(all_dict.items()):
        img_result=[] # 按照cls_dict的类别顺序来存放预测得到的str
        for label in cls_dict.values(): # 按照cls_dict的顺序来获取label
            label_str='' # 初始化为''
            for roi_name in label_dict[label]:
                roi_path=os.path.join(roi_folder,roi_name)
                roi_str=model.pred_roi(roi_path)
                label_str+=roi_str
            # 对于每一个label，进行pred_str的矫正，代码要放在这
            img_result.append(label_str)
        save_file.write(img_name+ ',' + ','.join(img_result) + '\n')
        print('\r{}/{} finished...'.format(idx + 1, len(all_imgs)), end=' ')
    save_file.flush()
    save_file.close()
    print('\nGOOD, All finished...')

import pandas as pd
import re
def change_final_predict_csv(file_path,save_path):
    def change_y(str1):
        str1 = str(str1)
        if len(str1) == 3:
            str1 = str1[0] + str1[1] + str1[1] + str1[2]
        if len(str1) == 5:
            if str1[2] == '0':
                str1 = str1[:2] + str1[3:]
            else:
                str1 = str1[:4]
        if str1.startswith('10'):
            str1 = '20' + str1[2:]
        if str1.startswith('29'):
            str1 = '19' + str1[2:]
        if str1.startswith('20') and int(str1[2]) > 2:
            str1 = '200' + str1[-1]
        if str1[:3] in ('193', '190'):
            str1 = '199' + str1[-1]
        elif str1[:3] in ('191', '194'):
            str1 = '197' + str1[-1]
        return int(str1)

    def change_d(str1):
        str1 = str(str1)
        if len(str1) == 3:
            str1 = str1[:2]
        if int(str1) == 0:
            str1 = '10'
        return int(str1)

    def change_add(str1):
        if re.match(r'^\D{2}市市\D{0,2}区', str1):
            str1 = re.sub(r'市市\D{0,2}区', '市市辖区', str1)
        if str1[-2] == '街':
            str1 = str1[:-1] + '道'
        str1 = re.sub(r'办\D{1}处', '办事处', str1)
        if str1.endswith('办事'):
            str1 += '处'
        elif str1[-3:-1] == '办事':
            str1 = str1[:-1] + '处'
        elif str1[-4:-1] == '街道办':
            str1 = str1[:-1] + '事处'
        elif str1[-5:-2] == '街道办':
            str1 = str1[:-2] + '事处'
        idx = str1.find('县级行政区')
        if idx >= 2:
            str1 = str1[:idx - 2] + '直辖县级行政区划' + str1[idx + 6:]
        return str1

    def change_range(str1):
        def change_y(y):
            if int(y) > 2020:
                y = '201' + y[3]
            elif int(y) < 2004:
                y = '20' + y[2:4]
            return y

        def change_md(m, fore='0'):
            if len(m) > 2:
                m = m[:2]
            elif len(m) == 1:
                m = fore + m
            return m

        str1 = str1.replace('年', '.').replace('月', '.')
        if '-长期' in str1:
            y, m, d = str1.split('.')
            y = change_y(y)
            m = change_md(m, '0')
            d = change_md(d, '1')
            return y + '.' + m + '.' + d + '-长期'
        else:
            first, second = str1.split('-')
            first_split = first.split('.')
            if len(first_split) == 3:
                y1, m1, d1 = first_split
            elif len(first_split) == 2:
                y1, m1 = first_split
                d1 = '01'
            second_split = second.split('.')
            if len(second_split) == 3:
                y2, m2, d2 = second_split
            elif len(second_split) == 2:
                y2, m2 = second_split
                d2 = '01'
            if len(y1) < 4:
                y1 = y1 + y2[(len(y1) - 4):]
            if len(y2) < 4:
                y2 = y2 + y1[(len(y2) - 4):]
            y1 = change_y(y1)
            # y2=change_y(y2)
            m1 = change_md(m1, '0')
            d1 = change_md(d1, '1')
            m2 = change_md(m2, '0')
            d2 = change_md(d2, '1')
            if y1 == y2 and 2004 <= int(y1) <= 2020:
                y2 = str(int(y1) + 20)
            if int(y2) - int(y1) not in (5, 10, 20):
                y2 = str(int(y1) + 20)
            if m1 != m2:
                m2 = m1
            if d1 != d2:
                d2 = d1
            return y1 + '.' + m1 + '.' + d1 + '-' + y2 + '.' + m2 + '.' + d2

    cols = ['name', 'minzu', 'sex', 'birth_y', 'birth_m', 'birth_d', 'address', 'id', 'unit', 'range']
    pred_df = pd.read_csv(file_path, header=None)
    pred_df.rename(columns=dict(zip(range(11), ['img_name'] + cols)), inplace=True)
    pred_df['name']=pred_df['name'].apply(lambda x: x+x if len(x)==1 else x) # minzu无需修改
    pred_df['birth_y']=pred_df['birth_y'].apply(change_y) # birth_m无需修改，
    pred_df['birth_d']=pred_df['birth_d'].apply(change_d)
    pred_df['address']=pred_df['address'].apply(change_add)
    pred_df['range']=pred_df['range'].apply(change_range)
    pred_df.to_csv(save_path,index=False)
    print('finished...')

from glob import glob
from collections import OrderedDict
import sys
def pred_folders_One_roi(roi_folder,result_save_path,roi_type,changed_save_path=None):
    cls_dict = {'name': 4, 'minzu': 3, 'sex': 6, 'birth': 1,
                'address': 0, 'id': 2, 'unit': 7, 'range': 5}
    label_type=str(cls_dict[roi_type.lower()])
    # cls=list(cls_dict.keys())
    all_names=set(os.listdir(roi_folder))
    all_imgs=np.unique([item.split('_')[0] for item in all_names if item.split('_')[2]==label_type])
    # 将所有的imgs整理好放在一个大的dict里面，key为img_name, value为一个list，放置对应的完整的img_name
    all_dict={}
    for img_name in all_imgs:
        temp_li=[]
        for i in [0,1]:
            for j in range(5):
                temp_name=img_name+'_'+str(i)+'_'+label_type+'_'+str(j)+'.jpg'
                if temp_name in all_names:
                    temp_li.append(temp_name)
        all_dict[img_name]=temp_li
    print('all_img_names prepared...,found {} imgs of {}'.format(len(all_dict),roi_type))
    ##### 测试一下看看是否正常：# 测试OK
    # num=0
    # for img_name,value_dict in all_dict.items():
    #     print(img_name,value_dict)
    #     num+=1
    #     if num>10: break
    # sys.exit(0)

    model = TechDing_One_Model(roi_type.lower())
    save_file = open(result_save_path, 'w', encoding='utf-8')
    if roi_type =='birth':
        save_file.write('img_name,birth_y,birth_m,birth_d'+'\n')
    else:
        save_file.write('img_name,'+roi_type.lower()+'\n')
    for idx,(img_name,rois_li) in enumerate(all_dict.items()):
        img_result=[]
        for roi_name in rois_li:
            roi_path=os.path.join(roi_folder,roi_name)
            roi_str=model.pred_roi(roi_path)
            img_result.append(roi_str)
        save_file.write(img_name+ ',' + ','.join(img_result) + '\n')
        print('\r{}/{} finished...'.format(idx + 1, len(all_imgs)), end=' ')
    save_file.flush()
    save_file.close()
    print('\nGOOD, All finished...')

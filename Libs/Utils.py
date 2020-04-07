# -*- coding: UTF-8 -*-

"""
# version: python3.6
# author: TechDing
# email: dingustb@126.com
# file: Utils.py
# time: 2019/9/11 12:47
# doc: 
"""

import pandas as pd
def print_right_ratio(true_csv_path,pred_csv_path):
    cols = ['Name', 'Minzu', 'Sex', 'Birth_Y', 'Birth_M', 'Birth_D', 'Address', 'ID', 'Unit', 'Range']
    true_df=pd.read_csv(true_csv_path,header=None)
    true_df.rename(columns=dict(zip(range(11), ['img_name']+['true_'+i for i in cols])), inplace=True)
    pred_df=pd.read_csv(pred_csv_path,header=None)
    pred_df.rename(columns=dict(zip(range(11), ['img_name'] + ['pred_' + i for i in cols])), inplace=True)
    merged_df = pd.merge(true_df, pred_df, how='inner', on='img_name')
    print('totally {} imgs.'.format(len(merged_df)))
    print('right ratios are: *****************************************************************')
    for col in cols:
        right_ratio = (merged_df['true_' + col] == merged_df['pred_' + col]).sum() / len(merged_df)
        print('{} : {}'.format(col, right_ratio))

import os
def save_err_samples(true_csv_path,pred_csv_path,err_df_save_folder):
    os.makedirs(err_df_save_folder,exist_ok=True)
    cols = ['Name', 'Minzu', 'Sex', 'Birth_Y', 'Birth_M', 'Birth_D', 'Address', 'ID', 'Unit', 'Range']
    true_df=pd.read_csv(true_csv_path,header=None)
    true_df.rename(columns=dict(zip(range(11), ['img_name']+['true_'+i for i in cols])), inplace=True)
    pred_df=pd.read_csv(pred_csv_path,header=None)
    pred_df.rename(columns=dict(zip(range(11), ['img_name'] + ['pred_' + i for i in cols])), inplace=True)
    merged_df = pd.merge(true_df, pred_df, how='inner', on='img_name')
    print('totally {} imgs.'.format(len(merged_df)))
    print('right ratios are: *****************************************************************')
    for col in cols:
        err_df = merged_df[merged_df['true_' + col] != merged_df['pred_' + col]]
        err_df[['img_name','true_' + col,'pred_' + col]].to_csv(os.path.join(err_df_save_folder,col+'_err.csv'),
                                                                index=None)
    print('finished...')


import pandas as pd
import numpy as np
from glob import glob
import os
def analyze_yolo_rois(rois_folder, labels_path):
    '''
    分析yolo预测出来的roi数量是否正确，此处仅仅预测roi的数量，而不管bbox的尺寸
    :param rois_folder: 所有YOLO预测出来的图片都存放在这个文件夹
    :param labels_path: 标准的train labels组成的csv
    :return:
    '''
    cls_dict = {'name': 4, 'minzu': 3, 'sex': 6, 'birth': 1,
                'address': 0, 'id': 2, 'unit': 7, 'range': 5}
    df_dict = {'address': 7, 'unit': 9}
    label_df = pd.read_csv(labels_path, header=None, index_col=[0])
    all_imgs = label_df.index.tolist()

    for img_name in all_imgs:
        for cls in list(cls_dict.keys()):
            cls_nums = len(glob(rois_folder + '/' + img_name + '_?_' + str(cls_dict[cls]) + '_*.jpg'))
            ok_num = 1 if cls in ['name', 'minzu', 'sex', 'birth', 'id', 'range'] else np.ceil(
                len(label_df.loc[img_name, df_dict[cls]]) / 12)
            if cls_nums != ok_num:
                print('{}: {} cls_len: {}, OK_len: {}'.format(cls, img_name, cls_nums, ok_num))
    print('finished...')

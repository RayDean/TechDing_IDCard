# -*- coding: UTF-8 -*-

"""
# version: python3.6
# author: TechDing
# email: dingustb@126.com
# file: others.py
# time: 2019/11/16 9:15
# doc: 
"""

import pandas as pd
import os,cv2
import numpy as np

def show_roi_change_label(roi_folder,raw_label_path,changed_label_save_path):
    '''
    将roi_folder中的图片显示出来，并手动修改对应的文本信息，如果正确，直接回车跳到下一张
    :param roi_folder: 原始的roi图片保存的文件夹
    :param raw_label_path: 原始的roi图片对应的label路径
    :param changed_label_save_path: 修改之后的roi保存的路径
    :return:
    '''

    raw_df=pd.read_csv(raw_label_path,encoding='utf-8',index_col=0)
    if not os.path.exists(os.path.dirname(changed_label_save_path)):
        os.makedirs(os.path.dirname(changed_label_save_path),exist_ok=True)

    all_rois=os.listdir(roi_folder)
    corrected=set()
    if os.path.exists(changed_label_save_path):
        with open(changed_label_save_path,'r',encoding='utf-8') as f:
            corrected=set([line.split(',')[0] for line in f.readlines()])
        file = open(changed_label_save_path, 'a', encoding='utf-8')
    else:
        file = open(changed_label_save_path, 'w', encoding='utf-8')
        file.write('img_name,label\n')
    need_delete=[]
    np.random.shuffle(all_rois) # 乱序
    for idx,roi_name in enumerate(all_rois):
        if roi_name in corrected: continue
        err_str=raw_df.ix[roi_name,'err']
        whole_str=raw_df.ix[roi_name,'predicted']
        if err_str in ('0_','5_'):
            file.write(roi_name + ',' + whole_str + '\n')
            continue
        roi=cv2.imread(os.path.join(roi_folder, roi_name))
        cv2.namedWindow('ROI', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('ROI',(218*2,76*2))
        cv2.imshow('ROI',roi)
        cv2.waitKey(1)
        put=input(whole_str+'  : '+err_str+'  ?')
        if put=='': # 如果直接回车，没有输入，表示正确
            file.write(roi_name+','+whole_str+'\n')
        elif put=='exit': # exit退出
            break
        elif put=='q': # quit不需要这张图片
            need_delete.append(roi_name)
            continue
        else:
            file.write(roi_name + ',' + put.strip() + '\n')
        file.flush()
        if (idx+1)%100==0:
            print('\r{}/{} finished...'.format(idx+1,len(all_rois)),end=' ')
            print()
    file.close()
    for delete in need_delete: # 删除quit的图片，防止下此再显示
        os.remove(os.path.join(roi_folder,delete))
    print('DONE!')


if __name__ == '__main__':
    roi_folder=r'F:\DataSet\IDCard\Round2\test2_ROIs\address'
    raw_label_path=r'F:\DataSet\IDCard\Unit_Analyze/test2_df_err4.csv'
    changed_label_save_path=r'F:\DataSet\IDCard\Unit_Analyze/test2_df_ok3.csv'
    show_roi_change_label(roi_folder, raw_label_path, changed_label_save_path)

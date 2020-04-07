# -*- coding: UTF-8 -*-

"""
# version: python3.6
# author: TechDing
# email: dingustb@126.com
# file: Main.py
# time: 2019/9/10 10:00
# doc: 
"""
from Libs import pred_folders_img
if __name__ == '__main__':
    # imgs_folder='E:\PyProjects\DataSet\OCR_IDCard\RawSet\Train'
    imgs_folder='E:\Test3'
    # result_save_path='E:\PyProjects\DataSet\OCR_IDCard\RawSet/Train_pred_V190910.csv'
    result_save_path='E:\Test3_result/Test_temp.csv'
    pred_folders_img(imgs_folder,result_save_path)

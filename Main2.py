# -*- coding: UTF-8 -*-

"""
# version: python3.6
# author: TechDing
# email: dingustb@126.com
# file: Main2.py
# time: 2019/9/10 17:35
# doc: 
"""


# 分阶段保存图片，分阶段查看是否有误
from keras.models import load_model
if __name__ == '__main__':
    pass
    ###########第一阶段，get_IDImg获取的imgs_folder中所有原始图像的正面反面图像并保存到save_folder中####################
    model=load_model('./Models/IDCard_UpDn_V191108_2') # OK
    print('extract model is loaded.')
    from Libs import get_IDImg
    imgs_folder=r'F:\DataSet\IDCard\Round2\train2'
    save_folder=r'F:\DataSet\IDCard\Round2\train2_IDcard2'
    import os
    all_imgs=os.listdir(imgs_folder)
    for idx, img_name in enumerate(all_imgs):
        get_IDImg(os.path.join(imgs_folder,img_name),model,save_folder)
        print('\r {}/{} finished...'.format(idx+1,len(all_imgs)),end=' ')

    print('finished..')


    #######################第二阶段：使用yolo来获取每张图片上的roi，并保存到文件夹#################
    # model=load_model('./Models/box_model_V190913')
    model=load_model(r'E:\PyProjects\DataSet\OCR_IDCard\Models/IDCard_yolo3_2_V190921.h5')
    # model=load_model(r'E:\PyProjects\DataSet\OCR_IDCard\Models/IDCard_OCR_V190829.h5') IDCard_OCR_V190829.h5
    print('box model is loaded.')
    from Libs import pred_roi
    imgs_folder=r'F:\DataSet\IDCard\Round2\train2_IDcard'
    save_folder=r'F:\DataSet\IDCard\Round2\train2_ROI_V0921'
    import os
    all_imgs=os.listdir(imgs_folder)
    for idx, img_name in enumerate(all_imgs):
        pred_roi(model,os.path.join(imgs_folder,img_name),save_folder)
        print('\r {}/{} finished...'.format(idx+1,len(all_imgs)),end=' ')

    print('finished..')


    ################################第三阶段：使用某模型来预测某种类别的roi，并保存结果到csv#########################
    from Libs import pred_folders_roi
    # roi_folder='E:\PyProjects\DataSet\OCR_IDCard\RawSet\Train_Card_Rois_V190921_4'
    roi_folder='E:\PyProjects\DataSet\OCR_IDCard\RawSet\Train_Card_Rois_V190829_4'
    # result_save_path='E:\PyProjects\DataSet\OCR_IDCard\RawSet/Train_Card_Rois_pred_V0916_2.txt'
    result_save_path='E:\PyProjects\DataSet\OCR_IDCard\RawSet/Train_Card_Rois_V0829_4_all2.txt'
    pred_folders_roi(roi_folder,result_save_path)


    ####################################测试################################################
    # from Libs import pred_folders_roi
    # # roi_folder = 'E:\PyProjects\DataSet\OCR_IDCard\RawSet\Train_Card_Rois_V190921'
    # roi_folder = 'E:\Test3_roi'
    # # result_save_path='E:\PyProjects\DataSet\OCR_IDCard\RawSet/Train_Card_Rois_pred_V0916_2.txt'
    # result_save_path = 'E:\Test3_roi/pred_test.txt'
    # pred_folders_roi(roi_folder, result_save_path)

    ##################仅仅预测某一种类别的结果#####################################################33
    # from Libs import pred_folders_One_roi
    # roi_folder = 'E:\PyProjects\DataSet\OCR_IDCard\RawSet\Train_Card_Rois_V190921_2'
    # # roi_folder = 'E:\PyProjects\DataSet\OCR_IDCard\RawSet\Train_Card_Rois_V190829_4'
    # # roi_folder = r'E:\PyProjects\DataSet\OCR_IDCard\RawSet\Train_Card_Pred_err_imgs_V0921_2\birth_y'
    # # result_save_path = 'E:\PyProjects\DataSet\OCR_IDCard\RawSet/Train_Card_Rois_pred_V0921_2_ID1.txt'
    # result_save_path = 'E:\PyProjects\DataSet\OCR_IDCard\RawSet/Train_Card_Rois_pred_V0921_2_name10307.txt'
    # roi_type='name'
    # pred_folders_One_roi(roi_folder, result_save_path,roi_type)
    #
    # #######################测试default_box#############################3
    # # model=load_model('./Models/box_model_V190913')
    # model=load_model(r'E:\PyProjects\DataSet\OCR_IDCard\Models/IDCard_yolo3_2_V190921.h5')
    # model=load_model(r'E:\PyProjects\DataSet\OCR_IDCard\Models/IDCard_OCR_V190829.h5')
    # print('box model is loaded.')
    # from Libs import pred_roi
    # imgs_folder='E:\Test5'
    # save_folder='E:\Test4\other_test_imgs'
    # import os
    # all_imgs=os.listdir(imgs_folder)
    # for idx, img_name in enumerate(all_imgs):
    #     pred_roi(model,os.path.join(imgs_folder,img_name),save_folder)
    #     print('\r {}/{} finished...'.format(idx+1,len(all_imgs)),end=' ')
    #
    # print('finished..')

    ###########################s使用extract_model来预测IDImg的类别##############################
    # import cv2
    # import numpy as np
    #
    # model = load_model('./Models/extract_model')
    # print('extract model is loaded.')
    # imgs_folder = r'F:\DataSet\IDCard\Round2\train2_IDcard'
    # save_path = r'F:\DataSet\IDCard\Round2\train2_IDcard.csv'
    #
    # import os
    # os.makedirs(os.path.dirname(save_path),exist_ok=True)
    # file=open(save_path,'w',encoding='utf-8')
    # file.write('name,label\n')
    # all_imgs = os.listdir(imgs_folder)
    # for idx, img_name in enumerate(all_imgs):
    #     img=cv2.imread(os.path.join(imgs_folder,img_name))
    #     label = np.argmax(model.predict(np.array([cv2.cvtColor(img[29:253, 112:336], cv2.COLOR_BGR2GRAY) \
    #                                                   [:, :, np.newaxis].astype(np.float32) / 255.0])), axis=1)[0]
    #     file.write(img_name+','+str(label)+'\n')
    #     print('\r {}/{} finished...'.format(idx + 1, len(all_imgs)), end=' ')
    #
    # print('finished..')

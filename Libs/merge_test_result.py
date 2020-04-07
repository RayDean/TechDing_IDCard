# -*- coding: UTF-8 -*-

"""
# version: python3.6
# author: TechDing
# email: dingustb@126.com
# file: merge_test_result.py
# time: 2019/11/13 16:25
# doc: 
"""

import os
import pandas as pd
import numpy as np

import re
def correct_birth(str1):
    y=re.findall(r'\d+年',str1)[0][:-1]
    if len(y)<4:# 绝大部分都是3位的错误
        y=y+y[-1]*(4-len(y))
    elif len(y)>4:
        y=y[:4]
    m=re.findall(r'\d+月',str1)[0][:-1]
    if len(m)>2: m=m[0]+m[-1]
    if int(m)>12: m='12'
    d=re.findall(r'\d+日',str1)[0][:-1]
    if len(d)>2: d=d[-2:] # 截取最后两位
    if int(d)>31: d='31'
    return y+'年'+m+'月'+d+'日'

def correct_range(str1):
    def change_y(y):
        y=y.replace('长','')
        if len(y)==3:
            y=y+y[-1]
        elif len(y)>4:
            y=y[:3]+y[-1]
        return y
    def change_m(m):
        if len(m) > 2:
            m = m[:2]
        elif len(m) == 1:
            m = m+m if m=='1' else  '03' if m=='0' else '0'+m
        return m
    def change_d(d):
        if len(d)>2:
            d=d[:2]
        elif len(d)==1:
            d=d+d if d in ('1','2') else '0'+d
        return d
    first,second=str1.split('-')[:2]
    if '长期' in second:
        y, m, d = str1.split('.')[:3]
        y = change_y(y)
        m = change_m(m)
        d = change_m(d)
        return y + '.' + m + '.' + d + '-长期'
    else:
        first_split = first.split('.')[:3]
        y1,m1=first_split[:2]
        d1=first_split[-1] if len(first_split)==3 else '01'
        second_split = second.split('.')[:3]
        y2, m2 = second_split[:2]
        d2=second_split[-1] if len(second_split)==3 else '01'
        y1 = change_y(y1)
        m1 = change_m(m1)
        d1 = change_d(d1)
        y2 = change_y(y2)
        m2 = change_m(m2)
        d2 = change_d(d2)
        if int(y2) - int(y1) not in (5, 10, 20):
            y2 = str(int(y1) + 10)
        if m1 != m2:
            m2 = m1
        if d1 != d2:
            d2 = d1
        return y1 + '.' + m1 + '.' + d1 + '-' + y2 + '.' + m2 + '.' + d2

def correct_name(str1):
    if len(str1)>3:str1=str1[:3]
    elif len(str1)==1:str1=str1+str1
    return str1


def load_dict(dict_save_path):
    with open(dict_save_path,'r',encoding='utf-8') as file:
        result=eval(file.read())
    return result

city_area_dict=load_dict(r'F:\DataSet\IDCard\Round2\dicts\city_area_dict.csv')

import Levenshtein,re
def correct_unit1(str1): # 这个最好
    def jaro_winkle_distance(str1, str2):
        sim = Levenshtein.jaro_winkler(str1, str2)
        return sim

    def search_area(city_area_dict, area):
        # 从city_area_dict中的area中查找area，如果找到返回对应的city，如果没找到，返回‘’
        city = ''
        for k, v in city_area_dict.items():
            if area in v:  # 找到area,
                city = k
                break
        return city

    def get_city_area(str1):
        ######排除掉公安局，分局等
        fenju = False
        if str1.endswith('分局'):
            fenju = True
            str1 = str1[:-2].replace('公安局', '')
        elif str1.endswith('安局'):
            str1 = str1[:-3]  # 排除掉最后的公安局三个字
        else:
            str1 = str1.replace('公安局', '').replace('分局', '')

        # 获取city, area
        city = ''
        area = ''
        if str1[0] == '县':
            city = '县'
            area = str1[1:]
            return city, area, fenju
        if str1[1:3] == '辖区':
            city = '市辖区'
            area = str1[3:]
            return city, area, fenju

        try:
            city = re.match(r'.+?(治州|市|地区|行政区划|盟)', str1).group(0)
            area = str1.replace(city, '', 1)
            return city, area, fenju
        except:
            if "自州" in str1:
                idx = str1.find("自州")
                city = str1[:idx] + '自治州'
                area = str1[(idx + 2):]
            elif "自治" in str1:
                idx = str1.find("自治")
                city = str1[:idx] + '自治州'
                area = str1[(idx + 2):]
            elif len(str1) < 6:
                city = '县'
                area = str1[1:]
            else:
                city = str1[:2] + '市'
                area = str1[2:]
            return city, area, fenju

    city, area, fenju = get_city_area(str1)
    ###########对市辖区进行矫正##############
    if city == '市辖区':
        # 先查找每个市辖区下面的area,看看哪一个对应上
        city_temp = ''
        for k, v in city_area_dict.items():
            if k.startswith('市辖区') and area in v:  # 找到area,
                city_temp = k
                break
        if city_temp == '':  # 没有找到
            part_list = []
            for k, v in city_area_dict.items():
                if k.startswith('市辖区'):
                    part_list.extend(list(v))
            sims = [jaro_winkle_distance(str1, area) for str1 in part_list]
            area = part_list[np.argmax(sims)]
        return city + '公安局' + area + '分局' if fenju else city + area + '公安局'
    ######对city, area进行矫正
    if city in city_area_dict.keys():  # city OK
        if area not in city_area_dict[city]:  # area wrong
            # 搜索全部的set是否有area
            find_city = search_area(city_area_dict, area)
            if find_city != '':  # 找到了
                city = find_city
            else:  # 没有找到
                # calc area, get max score area
                area_list = list(city_area_dict[city])
                area_sims = [jaro_winkle_distance(str1, area) for str1 in area_list]
                area = area_list[np.argmax(area_sims)]
        return city + '公安局' + area + '分局' if fenju else city + area + '公安局'
    else:  # city wrong
        # 搜索全部的set中是否包含有area,
        found = False
        for k, v in city_area_dict.items():
            if area in v:
                city = k
                found = True
                break
        #### 如果全部搜索都没有找到area，则计算city_area的相似度
        if not found:  # area wrong
            all_list = []
            for k, v in city_area_dict.items():
                all_list.extend([k + '|' + n for n in v])
            sims = [jaro_winkle_distance(str1, city + '|' + area) for str1 in all_list]
            city_area = all_list[np.argmax(sims)]
            city, area = city_area.split('|')
        return city + '公安局' + area + '分局' if fenju else city + area + '公安局'

def correct_unit2(str1): # 比unit1要差一些
    def jaro_winkle_distance(str1, str2):
        sim = Levenshtein.jaro_winkler(str1, str2)
        return sim

    def search_area(city_area_dict, area):
        # 从city_area_dict中的area中查找area，如果找到返回对应的city，如果没找到，返回‘’
        city = ''
        for k, v in city_area_dict.items():
            if area in v:  # 找到area,
                city = k
                break
        return city

    def get_city_area(str1):
        ######排除掉公安局，分局等
        fenju = False
        if str1.endswith('分局'):
            fenju = True
            str1 = str1[:-2].replace('公安局', '')
        elif str1.endswith('安局'):
            str1 = str1[:-3]  # 排除掉最后的公安局三个字
        else:
            str1 = str1.replace('公安局', '').replace('分局', '')

        # 获取city, area
        city = ''
        area = ''
        if str1[0] == '县':
            city = '县'
            area = str1[1:]
            return city, area, fenju
        if str1[1:3] == '辖区':
            city = '市辖区'
            area = str1[3:]
            return city, area, fenju

        try:
            city = re.match(r'.+?(治州|市|地区|行政区划|盟)', str1).group(0)
            area = str1.replace(city, '', 1)
            return city, area, fenju
        except:
            if "自州" in str1:
                idx = str1.find("自州")
                city = str1[:idx] + '自治州'
                area = str1[(idx + 2):]
            elif "自治" in str1:
                idx = str1.find("自治")
                city = str1[:idx] + '自治州'
                area = str1[(idx + 2):]
            elif len(str1) < 6:
                city = '县'
                area = str1[1:]
            else:
                city = str1[:2] + '市'
                area = str1[2:]
            return city, area, fenju

    city, area, fenju = get_city_area(str1)
    ###########对市辖区进行矫正##############
    if city == '市辖区':
        # 先查找每个市辖区下面的area,看看哪一个对应上
        city_temp = ''
        for k, v in city_area_dict.items():
            if k.startswith('市辖区') and area in v:  # 找到area,
                city_temp = k
                break
        if city_temp == '':  # 没有找到
            part_list = []
            for k, v in city_area_dict.items():
                if k.startswith('市辖区'):
                    part_list.extend(list(v))
            sims = [jaro_winkle_distance(str1, area) for str1 in part_list]
            area = part_list[np.argmax(sims)]
        return city + '公安局' + area + '分局' if fenju else city + area + '公安局'
    ######对city, area进行矫正
    if city in city_area_dict.keys():  # city OK
        if area not in city_area_dict[city]:  # area wrong
            # 搜索全部的set是否有area
            # find_city = search_area(city_area_dict, area)
            # if find_city != '':  # 找到了
            #     city = find_city
            # else:  # 没有找到
            # calc area, get max score area
            area_list = list(city_area_dict[city])
            area_sims = [jaro_winkle_distance(str1, area) for str1 in area_list]
            area = area_list[np.argmax(area_sims)]
        return city + '公安局' + area + '分局' if fenju else city + area + '公安局'
    else:  # city wrong
        # 搜索全部的set中是否包含有area,
        found = False
        for k, v in city_area_dict.items():
            if area in v:
                city = k
                found = True
                break
        #### 如果全部搜索都没有找到area，则计算city_area的相似度
        if not found:  # area wrong
            all_list = []
            for k, v in city_area_dict.items():
                all_list.extend([k + '|' + n for n in v])
            sims = [jaro_winkle_distance(str1, city + '|' + area) for str1 in all_list]
            city_area = all_list[np.argmax(sims)]
            city, area = city_area.split('|')
        return city + '公安局' + area + '分局' if fenju else city + area + '公安局'

def load_list(list_path):
    with open(list_path,'r',encoding='utf-8') as file:
        result=[line.strip() for line in file]
    return result
unit_lists=load_list(r'F:\DataSet\IDCard\Round2\dicts\city_area_list.csv')

def correct_unit_sims(str1): # 这个效果更差一些。
    def jaro_winkle_distance(str1, str2):
        sim = Levenshtein.jaro_winkler(str1, str2)
        return sim
    sims = [jaro_winkle_distance(raw, str1) for raw in unit_lists]
    return unit_lists[np.argmax(sims)]

def merge_test_result(params):
    '''
    将8中结果组合在一起，组成最终的结果。
    :param params:
        eg:
        params={'sex':r"F:\DataSet\IDCard\Round2\test2_ROIs\IDCard_Sex2_50W_V191110.csv",
            'minzu':r"F:\DataSet\IDCard\Round2\test2_ROIs\IDCard_Minzu2_50W_V191110.csv",
            'birth':r"F:\DataSet\IDCard\Round2\test2_ROIs\IDCard_Birth2_50W_V191111_base.csv",
            'id':r"F:\DataSet\IDCard\Round2\test2_ROIs\IDCard_ID2_50W_V191111_base.csv",
            'range':r"F:\DataSet\IDCard\Round2\test2_ROIs\IDCard_Range2_50W_V191111_base.csv",
            'name':r"F:\DataSet\IDCard\Round2\test2_ROIs\IDCard_Name2_100W_V1911111_base.csv",
            'unit':r"F:\DataSet\IDCard\Round2\test2_ROIs\IDCard_Unit2_100W_V191111_base.csv",
            'address':r"F:\DataSet\IDCard\Round2\test2_ROIs\IDCard_Address2_100W_V191112_base.csv",
            'save_path':r'F:\DataSet\IDCard\Round2\test2_ROIs\merged_V191113.csv'}
    :return:
    '''
    os.makedirs(os.path.dirname(params['save_path']),exist_ok=True)
    def load_df(kind):
        df=pd.read_csv(params[kind])
        df.columns=['img_name','pred']
        df['img_name']=df['img_name'].apply(lambda x: x[:-6])
        if kind=='birth':
            df['pred']=df['pred'].apply(correct_birth)
            df['y']=df['pred'].apply(lambda x: re.findall(r'\d+年',x)[0][:-1])
            df['m']=df['pred'].apply(lambda x: re.findall(r'\d+月',x)[0][:-1])
            df['d']=df['pred'].apply(lambda x: re.findall(r'\d+日',x)[0][:-1])
            df.drop(['pred'],axis=1,inplace=True)
        elif kind=='range':
            df['pred'] = df['pred'].apply(correct_range)
        elif kind=='name':
            df['pred']=df['pred'].apply(correct_name)
        elif kind=='unit':
            df['pred']=df['pred'].apply(correct_unit1)
        return df

    merged_df=None
    for kind in ['name','minzu','sex','birth','address','id','unit','range']:
        df=load_df(kind)
        merged_df=df if merged_df is None else pd.merge(merged_df,df,on='img_name')
    merged_df.to_csv(params['save_path'],header=None,index=False)
    print('finished...')

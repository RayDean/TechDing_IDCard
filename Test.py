# -*- coding: UTF-8 -*-

"""
# version: python3.6
# author: TechDing
# email: dingustb@126.com
# file: Test.py
# time: 2019/11/13 16:50
# doc: 
"""

if __name__ == '__main__':
    from  Libs import merge_test_result
    params = {'sex': r"F:\DataSet\IDCard\Round2\test2_ROIs\IDCard_Sex2_50W_V191110.csv",
              'minzu': r"F:\DataSet\IDCard\Round2\test2_ROIs\IDCard_Minzu2_50W_V191110.csv",
              'birth': r"F:\DataSet\IDCard\Round2\test2_ROIs\IDCard_Birth2_50W_V191111_base.csv",
              'id': r"F:\DataSet\IDCard\Round2\test2_ROIs\IDCard_ID2_50W_V191111_base.csv",
              'range': r"F:\DataSet\IDCard\Round2\test2_ROIs\IDCard_Range2_50W_V191111_base.csv",
              'name': r"F:\DataSet\IDCard\Round2\test2_ROIs\IDCard_Name2_100W_V1911111_base.csv",
              'unit': r"F:\DataSet\IDCard\Round2\test2_ROIs\IDCard_Unit2_100W_V191111_base.csv",
              # 'unit': r"F:\DataSet\IDCard\Round2\test2_ROIs\IDCard_Unit0_100W_V191113_base.csv",
              # 'address': r"F:\DataSet\IDCard\Round2\test2_ROIs\IDCard_Address2_100W_V191112_base.csv",
              'address': r"F:\DataSet\IDCard\Round2\test2_ROIs\IDCard_Address0_100W_V191113_base.csv",
              'save_path': r'F:\DataSet\IDCard\Round2\test2_ROIs\merged_V191115_2.csv'}
    merge_test_result(params)

# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 17:38:38 2017

@author: brucedai
"""

import os
from feature import timer_decorator
import pandas as pd
import numpy as np

output_dir = './实时收盘价/'

@timer_decorator
def seperateDataset():
    '''
    从万德下载下来的原始数据中分离出每一个指数的数据，并存入相应的csv文件。
    '''
    df = pd.read_excel('指数数据.xlsx')
    # 去掉重复的index
    df.index = pd.Index(data=df.index.levels[0])
    df.index.name = 'date_time'
    
    col = df.columns
    ind = range(int(len(col)/5))
    col_names = ['close', 'volume']
    
    
    
    for i in ind:
        
        print('当前处理第%d个指数数据, 剩余%d个' % (i+1, ind[-1] - i))
        # 指数名称
        index_name = col[i*5].split('.')[0]
        col_i = col[i*5+3 : (i+1)*5]
        df_i = df[col_i]
        # 重新设置列名称
        df_i.columns = col_names
        # 保存df
        f_path = os.path.join(output_dir, '%s.csv' % index_name)
        df_i.to_csv(f_path)
        

def transformPrimaryDataset(name=['上证50', '沪深300', '中证500']):
    '''
    从主要关注的指数（且具有完整数据）数据文件中进一步提取数据，计算在14：56
    的收盘价和当天累计成交量。
    '''
    df_names = pd.read_csv('指数名称.csv', dtype=np.str, engine='python')
    df_names.set_index('code', inplace=True)
    # 得到相应的指数代码名称
    code = df_names.index[np.isin(df_names['name'], name)].tolist()
    # 读取该指数14:56分的数据
    for c in code:
        path_old = os.path.join('./dataset', '%s.csv' % c)
        path_new = os.path.join(output_dir, '%s.csv' % c)
        df_old = pd.read_csv(path_old, index_col='date', parse_dates=True)
        df_new = pd.read_csv(path_new, index_col='date_time', parse_dates=True,
                             engine='python')




if __name__ == '__main__':
    
    seperateDataset()
    
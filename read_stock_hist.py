# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 10:43:48 2017

@author: brucedai
Content: 利用tushare模块读取26个股票指数的历史数据，并保存。

"""

import os
from time import time
import pandas as pd
import tushare as ts


def timer_decorator(func):
    
    def wrapper(*args):
        print('-'*50)
        t1 = time()
        func(*args)
        t2 = time()
        t_diff = t2 - t1
        print('耗时为%.2f秒' % t_diff)
        
    return wrapper

@timer_decorator
def download_save(row, output_dir):
    
    # 1. 下载最新数据
    columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    data = ts.get_k_data(code=row.code, start='2017-08-01', index=True)
    data.drop('code', axis=1, inplace=True)
    data = data.reindex(columns=columns)
    data.date = pd.to_datetime(data.date)
    # 下载下来的数据volume单位是手，而历史数据中volume单位是股，所以需要x100
    data.volume = data.volume * 100
    print('%s\t下载%s数据完毕，代码：%s' %(row.name+1, row['name'], row.code))
    
    # 2. 读取原先的数据，进行合并处理
    data_old = pd.read_csv('newdata/%s.csv' % row.code)
    data_old.date = pd.to_datetime(data_old.date)
    data_old = data_old.reindex(columns=columns)
    
    # 3. 合并数据集
    date = pd.to_datetime('2017-09-10')
    data_new = pd.concat([data_old.loc[data_old.date < date, :], 
                          data.loc[data.date >= date, :]], ignore_index=True)

    # 4. 保存数据集
    path = os.path.join(output_dir, '%s.csv' % row.code)
    data_new.to_csv(path, index=False)
    print('数据合并完毕')


@timer_decorator
def main():
    output_dir = './newdata'
    index = ts.get_index()
    for _, row in index.iterrows():
        download_save(row, output_dir)
        
        
if __name__ == '__main__':
    main()
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 10:04:01 2017

@author: brucedai
"""
import os
from time import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

data_dir = './dataset/'
output_dir = './stats/'

def ROCP(f_name, df):
    '''
    读取f_name中的股票/期货行情数据，计算涨跌幅，并存储到data中
    '''
    print('读取%s文件数据:' % f_name)
    data = pd.read_csv(os.path.join(data_dir, f_name), sep='\t')
    ret = data.close.pct_change()
    col = f_name.split(sep='.')[0]
    df[col] = ret
    return df


def Plot_and_Savefig():
    df = pd.read_csv(os.path.join(output_dir, 'stocks.csv'), index_col=0)
    for col in df:
        data = df[col].dropna()
        mean_data = data.mean()
        std_data = data.std()
        skew_data = data.skew()
        kurt_data = data.kurt()
        print('股票%s日涨跌幅统计数据:' % col)
        print('共%d个数据' % data.shape[0])
        print('均值:\t%.4f' % mean_data)
        print('标准差:\t%.4f' % std_data)
        print('偏度:\t%.4f' % skew_data)
        print('峰度:\t%.4f' % kurt_data)
        
        fig, ax = plt.subplots(1, 1)
        # 画出相应的正态分布
        x_data = np.linspace(norm.ppf(0.0001,
                                      loc=data.mean(), scale=data.std()),
                             norm.ppf(0.9999,
                                      loc=data.mean(), scale=data.std()), 1000)
        y_data = norm.pdf(x_data, loc=data.mean(), scale=data.std())
        ax.plot(x_data, y_data, 'r-', lw=2, alpha=0.6, label='正态分布')
        ax.hist(data, bins=50, normed=True, histtype='stepfilled', alpha=0.3)
        plt.title('股票%s日涨跌幅统计直方图' % col)
        plt.savefig(os.path.join(output_dir, '%s.png' % col))
        

def Extract_ROCP():
    '''
    读取data_dir里所有行情数据，计算涨跌幅后保存到output_dir一个文件中。
    '''
    start_time = time()
    file_list = os.listdir(data_dir)
    column_names = [s.split(sep='.')[0] for s in file_list]
    df = pd.DataFrame(data=None, columns=column_names)
    for f in file_list:
        df = ROCP(f, df)
    read_size = df.size
    diff_time = time() - start_time
    print('程序读取涨跌幅数据总量为%d\n耗时%.2f秒' %
          (read_size, diff_time))
    
    if not(os.path.exists(output_dir)):
        os.mkdir(output_dir)
    
    df.to_csv(os.path.join(output_dir, 'stocks.csv'))
    diff_time = time() - start_time
    print('程序存储涨跌幅数据总量为%d\n耗时%.2f秒' %
          (read_size, diff_time))
    
    
if __name__ == '__main__':
    Plot_and_Savefig()
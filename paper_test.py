# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 17:58:53 2017

@author: brucedai
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import pandas as pd
import numpy as np

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from feature import extract_from_file, extract_all_features
from dataset import DataSet
from windpuller import WindPuller


# 全局变量
model_path_buy = 'model.30.buy'
model_path_sell = 'model.30.sell'
data_dir = './newdata/'
tsl_data_dir = './tsl_pre_data'
feature_dir = './stock_features/'
input_shape = [30, 61]


def set_gpu_fraction():
    '''
    设置每一个keras模型使用20%的GPU显存。
    '''
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    set_session(tf.Session(config=config))


def get_date_list():
    '''
    获取8/1之后的交易日期列表
    '''
    data = pd.read_csv('./tsl_pre_data/000001.csv')
    date = data['date'].tolist()
    return date


def read_features(path, input_shape, prefix):
    '''
    只读取测试集的数据
    '''
    
    test_features = np.load("%s/%s_feature.test.%s.npy" %
                            (path, prefix, str(input_shape[0])))
    test_features = np.reshape(test_features,
                               [-1, input_shape[0], input_shape[1]])
    test_labels = np.load("%s/%s_label.test.%s.npy" % (path, prefix,
                                                       str(input_shape[0])))
    # test_labels = np.reshape(test_labels, [-1, 1])
    test_set = DataSet(test_features, test_labels)
    return test_set


def simple_predict_tomorrow():
    '''
    使用做多和做空两个模型，对data_dir里的每日行情数据提取特征并训练得到信号。
    '''

    signal_dir = './signal_close/'
    date = get_date_list()
    files = os.listdir(data_dir)
    
    # 0. 加载模型
    wp_buy = WindPuller(input_shape).load_model(model_path_buy)    
    wp_sell = WindPuller(input_shape).load_model(model_path_sell)
    
    # 1. 提取所有特征
    days_for_test = len(date)
    extract_all_features(data_dir, feature_dir, days_for_test)
    
    for (idf, f) in enumerate(files):
                
        # 2. 读取测试集特征
        output_prefix = f.split('.')[0]
        test_set = read_features(feature_dir, input_shape, output_prefix)
    
        # 3. 训练模型
        signal_buy = wp_buy.predict(test_set.images, 1024)
        signal_buy = signal_buy[-days_for_test:]
        
        signal_sell = wp_sell.predict(test_set.images, 1024)
        signal_sell = signal_sell[-days_for_test:]
        
        # 4. 保存结果
        f_path_signal = os.path.join(signal_dir, f)
        data_signal = pd.DataFrame({'signal_close_buy': signal_buy.reshape(-1),
                                    'signal_close_sell': signal_sell.reshape(-1)},
                                   index = date)
        data_signal.to_csv(f_path_signal)
        print('%d 指数%s处理完毕' % (idf, output_prefix))
        print('-' * 50)
        
    print('全部处理完毕！')
    print('=' * 80)
        

def paper_test():
    '''
    逐个读取每一天的14：57的数据，与数据库中数据合并，生成新特征，读取训练好的模型，
    预测出信号。
    ''' 
    merged_data_dir = './paper_merge'
    signal_dir = './paper_signals'
    date = get_date_list()
    files = os.listdir(tsl_data_dir)

        
    # 0. 加载模型
    wp_buy = WindPuller(input_shape).load_model(model_path_buy)
    wp_sell = WindPuller(input_shape).load_model(model_path_sell)
    
    for (idx, d) in enumerate(date):
        
        print('当前处理日期\t%s' % d)
        for (idf, f) in enumerate(files):
            
            # 1. 读取新的数据
            f_path1 = os.path.join(tsl_data_dir, f)
            df1 = pd.read_csv(f_path1)
            # 获取某一天的数据
            df1 = df1[df1['date'] == d]
            df1['volume'] == df1['volume'] * 80/79
            
            # 2. 读取原来的数据
            f_path2 = os.path.join(data_dir, f)
            df2 = pd.read_csv(f_path2)
            
            # 3. 合并数据，删除原来数据多余部分，追加最新的一天的数据
            df2 = df2.iloc[: int(np.flatnonzero(df2.date == d))]
            df3 = df2.append(df1, ignore_index=True)
            df3 = df3[df2.columns]
            
            # 4. 保存数据
            f_path_merged = os.path.join(merged_data_dir, f)
            df3.to_csv(f_path_merged, index=False)
            
            # 5. 提取1个特征，存入相应文件夹
            output_prefix = f.split('.')[0]
            extract_from_file(idx, f_path_merged, feature_dir,
                              output_prefix, 1)
            
            # 6. 读取提取完的特征
            test_set = read_features(feature_dir, input_shape, output_prefix)
            
            # 7. 训练模型
            signal_buy = wp_buy.predict(test_set.images, 1024)
            signal_buy = float(signal_buy[-1])
            
            signal_sell = wp_sell.predict(test_set.images, 1024)
            signal_sell = float(signal_sell[-1])
            
            # 8. 保存结果
            f_path_signal = os.path.join(signal_dir, f)
            
            if idx == 0:
                # 写入字段名
                title = 'date,signal_buy,signal_sell'
                with open(f_path_signal, 'a') as file:
                    file.write(title)
            
            write = '%s,%.2f,%.2f\n' %(d, signal_buy, signal_sell)            
            with open(f_path_signal, 'a') as file:
                file.write(write)
            
            n_read = idx * len(files) + idf + 1
            print('当前处理第%d个文件，剩余%d个文件，请耐心等待...' %
                  (n_read, len(files) * len(date) - n_read))
            print('-'*50)
            
    print('\n全部处理完毕！')
    print('='*80)
 
    
def calc_strat_perf():
    '''
    计算实际涨跌幅，以及策略的资金曲线。
    '''
    sig_dir = './合并后的信号/'
    date = get_date_list()
    files = os.listdir(data_dir)
    
    for (idf, f) in enumerate(files):
        
        print('%d当前处理%s文件\n' % (idf+1, f))

        # 1. 读取行情数据
        f_path = os.path.join(data_dir, f)
        df_close = pd.read_csv(f_path, index_col='date', parse_dates=True)
        df_close = df_close.iloc[df_close.index.get_loc(date[0]):, :]

        # 2. 读取信号数据
        sig_path = os.path.join(sig_dir, f)
        df_sig = pd.read_csv(sig_path, index_col='date', parse_dates=True)
        
 
        df_sig = df_sig.assign(close=df_close['close'])
        # 计算实际涨跌幅
        df_sig['pct_change'] = df_sig['close'].pct_change() * 100
        df_sig['pct_change'] = df_sig['pct_change'].apply(
                                lambda x: np.round(x, 2))
       
        # 3. 保存结果至源文件
        df_sig.to_csv(sig_path)
        
    print('全部处理完毕！')
    print('=' * 80)    


def generate_3_index_signals():
    '''
    生成回测数据中特定的3个指数的信号文件
    '''
    code_list = ['000016', '000300', '000905']
    sig_dir = './合并后的信号'
    dst_dir = '3个指数信号'
    for (idx, code) in enumerate(code_list):
        
        print('%d当前处理%s文件\n' % (idx+1, code))
        
        # 1. 读取信号源文件
        f = code + '.csv'
        f_path = os.path.join(sig_dir, f)
        data = pd.read_csv(f_path, parse_dates=True, index_col='date')
        
        # 2. 生成相应数据
        if code == '000016':
            data = data[['signal_close_buy', 'signal_buy', 'diff_buy',
                         'close', 'pct_change']]
        elif code == '000300':
            data = data[['signal_close_buy', 'signal_buy', 'diff_buy',
                         'signal_close_sell', 'signal_sell', 'diff_sell',
                         'close', 'pct_change']]
        elif code == '000905':
            data['signal_close_buy_screened'] = \
            np.where(data['signal_close_sell'] < 0,
                     0, data['signal_close_buy'])
            data['signal_buy_screened'] = \
            np.where(data['signal_sell'] < 0, 0, data['signal_buy'])
            data['signal_close_buy_screened'] = \
            data['signal_close_buy_screened'].apply(lambda x: np.round(x, 2))
            data['signal_buy_screened'] = \
            data['signal_buy_screened'].apply(lambda x: np.round(x, 2)) 
            data['diff_buy_screened'] = \
            (data['signal_close_buy_screened'] != \
                           data['signal_buy_screened']).astype(np.int)
            data = data[['signal_close_buy_screened', 'signal_buy_screened',
                         'diff_buy_screened',
                         'close', 'pct_change']]

        # 3. 存储
        d_path = os.path.join(dst_dir, f)
        data.to_csv(d_path)
    
    print('全部处理完毕！')
    print('=' * 80)    

    
def correct_signals():
    '''
    修改paper_test之前写错，写入的数据
    注：很少用到的函数
    2017-09-17 17:27
    '''
    signal_dir = './paper_signals'
    
    files = os.listdir(signal_dir)
    
    for (idf, f) in enumerate(files):
        
        print('%d当前处理%s文件\n' % (idf+1, f))
        f_path = os.path.join(signal_dir, f)
        
        data = pd.read_csv(f_path, skiprows=30, header=0, 
                           names=['a', 'b', 'date', 'signal_buy',
                                  'signal_sell'])
        # 只保留后3列数据
        data.drop(['a', 'b'], axis=1, inplace=True)
        # 修改date一列连在一起的数据
        data['date'] = data['date'].apply(lambda x: x[12:])
        # 去掉signal_sell末尾误加的f
        data.signal_sell = data.signal_sell.apply(lambda x:x[:-1])
        
        # 保存数据
        data.to_csv(f_path, index=False)
        
    print('全部处理完毕！')
    print('=' * 80)


def merge_signals_buy():
    '''
    合并做多和做空模型的信号，生成对比文件。
    '''
    sig1_dir = './signal_close'
    sig2_dir = './paper_signals'
    sig3_dir = './合并后的信号'
    
    files = os.listdir(tsl_data_dir)
    
    for (idf, f) in enumerate(files):
        
        print('%d\t当前处理%s的信号合并' % (idf+1, f))
        
        # 日收盘信号
        f1_path = os.path.join(sig1_dir, f)
        df1 = pd.read_csv(f1_path, index_col=0)
        df1.index.name = 'date'
        
        # 提前结束信号
        f2_path = os.path.join(sig2_dir, f)
        df2 = pd.read_csv(f2_path)
        df2.set_index('date', inplace=True)
        
        # 合并
        df3 = df1.merge(df2, on=None, left_index=True, right_index=True)
        df3['diff_buy'] = (df3['signal_close_buy'] != \
                           df3['signal_buy']).astype(np.int)
        df3['diff_sell'] = (df3['signal_close_sell'] != \
                            df3['signal_sell']).astype(np.int)
        # 修改为2位小数
        df3[['signal_close_buy', 'signal_close_sell']] = \
            df3[['signal_close_buy', 'signal_close_sell']].apply(
                    lambda x: np.round(x, 2))
        
        f2_path = os.path.join(sig3_dir, f)
        df3.to_csv(f2_path)
        
    print('全部处理完毕！')
    print('='*50)




if __name__ == '__main__':
    set_gpu_fraction()
    simple_predict_tomorrow()
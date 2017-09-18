# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 11:14:16 2017

@author: brucedai





"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
from datetime import datetime, timedelta
from dateutil.parser import parse

import pandas as pd
import numpy as np
import tushare as ts
from apscheduler.schedulers.background import BackgroundScheduler

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from dataset import DataSet
from windpuller import WindPuller
from feature import extract_all_features

# 全局变量
name_dict = {'中证500': '000905', '上证50': '000016', '沪深300': '000300'}
data_dir = './newdata/'
feature_dir = './stock_features/'
compare_dir='./compare_data'
input_shape = [30, 61]


def set_gpu_fraction():
    
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    set_session(tf.Session(config=config))


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


def get_realtime_data():
    '''
    获取中证500,上证50,沪深300的实时行情
    '''
    running = 1
    data = ts.get_realtime_quotes(['zh500', 'sz50', 'hs300'])
    data = data.apply(pd.to_numeric, errors='ignore')
    data[['open', 'high', 'low', 'price', 'volume']] = \
        data[['open', 'high', 'low', 'price', 'volume']].astype(float)
    # 实时行情数据的最新时间戳，datetime类型
    time = parse(data['date'][0] + ' ' + data['time'][0])
    if time >= datetime.today().replace(hour=14,
                            minute=57, second=0, microsecond=0):
        # 一旦获取到的时间戳超过当日14:57，则停止继续查询
        running = 0
        
    return running, data
    
    
def update_csv(data):
    '''
    更新本地行情数据。
    data是14:56的数据，先对成交量进行校正，然后更新到本地csv数据文件中。
    '''
    
    # 预处理data
    data.rename(columns={'price': 'close'}, inplace=True)
    data['volume'] = np.round(data['volume'] * 100 * 80/79)
    data = data[['date', 'open', 'high', 'low', 'close', 'volume', 'name']]
    data.set_index(keys='date', inplace=True)
    data.index = pd.to_datetime(data.index)
    
    for name in name_dict:
        
        # 添加数据到末尾
        code = name_dict[name]
        f_path = os.path.join(data_dir, '%s.csv' % code)
        df_old = pd.read_csv(f_path, index_col=['date'], parse_dates=True,
                             engine='python')
        df_old.index = pd.to_datetime(df_old.index)
        if df_old.index[-1] != data.index[-1]:
            mask = data['name'] == name
            df_old = df_old.append(data[mask].drop('name',axis=1))
            print('已成功添加%s的%s数据\n' % (
                    df_old.index[-1].strftime('%Y-%m-%d'), name))
            # 保存数据，替换源文件
            df_old.to_csv(f_path)   
            
            # 添加一行数据到用来比较的文件中
            # 日期，盘中价格
            write = '%s,%s,' % (data.index[0].strftime('%Y-%m-%d'),
                                     float(data[mask]['close']))
            
            compare_path = os.path.join(compare_dir, '%s.csv' % code)
            with open(compare_path, 'a') as f:
                f.write(write)
        else:
            print('无需更新，%s的%s数据已存在！\n' % (
                  df_old.index[-1].strftime('%Y-%m-%d'), name))
    
    print('盘中数据合并结束\n')
    print('=' * 80)
    

def update_close_csv(data):
    '''
    收盘后的价格和成交量的数据更新
    '''
 
    # 实时行情数据的最新时间戳，datetime类型
    time = parse(data['date'][0] + ' ' + data['time'][0])
    print('更新今日收盘价数据%s' % time)
    
    # 预处理data
    data.rename(columns={'price': 'close'}, inplace=True)
    data['volume'] = data['volume'] * 100
    data = data[['date', 'open', 'high', 'low', 'close', 'volume', 'name']]
    data.set_index(keys='date', inplace=True)
    data.index = pd.to_datetime(data.index)
    
    for name in name_dict:
        
        code = name_dict[name]
        f_path = os.path.join(data_dir, '%s.csv' % code)
        df_old = pd.read_csv(f_path, index_col=['date'], parse_dates=True,
                             engine='python')
        df_old.index = pd.to_datetime(df_old.index)
        
        if df_old.index[-1] == data.index[-1]:
            mask = data['name'] == name
            # 删除原来的最后一行数据
            df_old = df_old[:-1]
            df_old = df_old.append(data[mask].drop('name',axis=1))
            print('已成功修改%s的%s数据\n'  % (
                    df_old.index[-1].strftime('%Y-%m-%d'), name))
            # 保存数据，替换源文件
            df_old.to_csv(f_path)
            
            # 添加数据到用来比较的文件中
            # 日收盘价
            write = '%s,' % float(data[mask]['close'])
            
            compare_path = os.path.join(compare_dir, '%s.csv' % code)
            with open(compare_path, 'a') as f:
                f.write(write)
                
        else:
            print('无需更新!\n')
    print('收盘数据合并结束\n')
    print('=' * 80)

def write_new_signal(output_dir, compare_dir='./compare_data'):
    '''
    将最新的信号写入信号对比文件中
    '''
    
    for name in name_dict:
        
        code = name_dict[name]
        # 添加一行数据到用来比较的文件中
        # 先获取原先预测的信号
        signal_path = os.path.join(output_dir, 'stocks',
                                   '%s_test.csv' % name)
        signal_df = pd.read_csv(signal_path, index_col='date',
                                engine='python')
        signal = signal_df.iloc[-1]['Position']
        
        # 再追加到文件末尾
        write = '%s\n' % (signal)
        
        compare_path = os.path.join(compare_dir, '%s.csv' % code)
        with open(compare_path, 'a') as f:
            f.write(write)

def predict_tomorrow(wp_buy, wp_sell, is_last_column):
    '''
    根据给定的做多和做空模型，预测明天的仓位信号，打印出来，并保存结果。
    '''
    titles = ['品种', '时间', '做多信号', '做空信号', '过滤信号']
    print('{:>12} {:>12} {:>12} {:>12} {:>12}'.format(*titles))
    
    for name in name_dict:
        
        # 1. 读取测试集特征数据
        code = name_dict[name]
        test_set = read_features(feature_dir, input_shape, code)
        
        # 2. 预测买入和卖出信号
        signal_buy = float(wp_buy.predict(test_set.images, 1024))
        signal_sell = float(wp_sell.predict(test_set.images, 1024))
        
        # 3. 显示时间和信号
        # 计算过滤后的做多信号
        if signal_sell != 0:
            signal_screened = 0.0
        else:
            signal_screened = signal_buy
        
        print_time = datetime.now().strftime('%H:%M:%S')
        print('{:>12} {:>12} {:>12.2f} {:>12.2f} {:>12.2f}'.format(name,
                                            print_time,
                                            signal_buy, signal_sell,
                                            signal_screened))
        
        # 4. 写入到文件中
        # 信号记录时间，做多信号，做空信号，过滤信号
        if is_last_column:
            formatter = '\n'
        else:
            formatter = ','
            
        # 需要根据指数代码，判断写入哪些数据
        if code == '000016':
            # 上证50
            write = ('%.2f' + formatter) % signal_buy
        elif code == '000300':
            # 沪深300
            write = ('%.2f,%.2f' + formatter) % (signal_buy, signal_sell)
        elif code == '000905':
            # 中证500
            write = ('%.2f' + formatter) % signal_screened

        compare_path = os.path.join(compare_dir, '%s.csv' % code)
        with open(compare_path, 'a') as f:
            f.write(write)        
        
    print('\n', '-'*25, ' 全部信号记录完毕 ', '-'*25, '\n') 
    print('=' * 80)
    

def main():
    '''
    每天获取14:57的3个指数数据，进行数据校正后添加到本地数据文件中，再提取特征，计算
    信号，保存到对应文件中。到15:01收盘后，再进行同样操作。
    '''
    # 1. 加载keras训练完的模型
    print('='*80)
    print('%s\t加载keras训练完的模型' % (datetime.now().strftime('%H:%M:%S')))
    set_gpu_fraction()
    model_path_buy = 'model.30.buy'
    wp_buy = WindPuller(input_shape).load_model(model_path_buy)
    
    model_path_sell = 'model.30.sell'
    wp_sell = WindPuller(input_shape).load_model(model_path_sell)
    print('\n%s\t模型加载完毕\n' % (datetime.now().strftime('%H:%M:%S')))    
    
    
    # 2. 查询14:57的数据
    print('='*80)
    print('%s\t开始查询实时行情数据，将返回14:57的第一笔数据' %
          datetime.now().strftime('%H:%M:%S'))
    running = 1
    # 程序最多查询到15:01
    stop_time = datetime.now().replace(hour=15,
                            minute=1, second=0, microsecond=0)
    
    while running:
        
        print('时间未到，请耐心等待数据...')
        running, data = get_realtime_data()
        if running == 1:
            time.sleep(3)
        # 只要获取到14:57的数据，或者查询时间超过15:01，就停止
        running = running and datetime.now() < stop_time
    
    print('%s\t查询数据完毕，开始合并数据\n' %
          datetime.now().strftime('%H:%M:%S'))
    print('='*80)     
    
    # 3. 更新本地数据，共2个文件
    # 文件1：原始数据末尾添加一行14:56数据
    # 文件2：对比信号文件添加1行4列数据
    update_csv(data)
    
    # 4. 提取最新特征
    print('%s\t开始提取特征\n' % datetime.now().strftime('%H:%M:%S'))
    extract_all_features(data_dir, feature_dir, days_for_test=1,
                         extract_all=False)
    print('%s\t特征提取完毕\n' % datetime.now().strftime('%H:%M:%S'))
    print('='*80)       
    
    # 5. 读取原始数据，生成特征，预测明天的信号，并保存
    predict_tomorrow(wp_buy, wp_sell, is_last_column=False)
    print('请等待15:01程序会继续获取当日行情数据进行计算和预测...')
    print('=' * 80)
    
    # 6. 在15:01:20运行一次, 获取收盘后的行情数据
    stop_time = stop_time.replace(second=20)
    while datetime.now() < stop_time:
        time.sleep(3)
        print('等待中，请勿中断...')
       
    print('='*25, '开始获取当日收盘后行情', '='*25)
    _, data = get_realtime_data()
    
    # 7. 使用当日收盘价更新本地数据文件，共2个文件
    # 文件1：原始数据末尾修改14:57的数据
    # 文件2: 对比信号文件最后一行追加2列数据
    update_close_csv(data)

    # 8. 提取最新特征
    print('%s\t开始提取特征\n' % datetime.now().strftime('%H:%M:%S'))
    extract_all_features(data_dir, feature_dir, days_for_test=1,
                         extract_all=False)
    print('%s\t特征提取完毕\n' % datetime.now().strftime('%H:%M:%S'))
    print('='*80)       
    
    # 9. 读取收盘后的原始数据，生成特征，预测明天的信号
    predict_tomorrow(wp_buy, wp_sell, is_last_column=True)

    print('%s\t完成!\n' % datetime.now().strftime('%H:%M:%S'))
    print('=' * 80)
    

def test_timer():
    '''
    测试通过datetime模块获取时间来控制程序的执行
    '''
    running = 1
    stop_time = datetime.now() + timedelta(seconds=60)
    while running:
        
        print('I\'m working...')
        time.sleep(5)
        running = datetime.now() < stop_time
        
    print('I\'m retired!')


def test():
    '''
    测试apscheduler每隔10秒获取一次最新行情，存储到本地文件并对比时间戳。
    '''
    # 写入文件的header
    with open('logScheduler.csv', 'a') as f:
        f.writelines('系统时间,数据时间,中证500价格,上证50价格,沪深300价格\n')
    print('系统时间,数据时间,中证500价格,上证50价格,沪深300价格')
    
    sched = BackgroundScheduler(standalone=True)
    def log():

        # 获取中证500数据
        data = ts.get_realtime_quotes(['zh500', 'sz50', 'hs300'])
        time = data['time'][0]
        time_sys = datetime.now().strftime('%H:%M:%S')
        price = data['price']
        with open('logScheduler.csv', 'a') as f:
            f.writelines('%s,%s,%f,%f,%f\n' %(time_sys, time, float(price[0]),
                                              float(price[1]),
                                              float(price[2])))
        print('%s,%s,%f,%f,%f' %(time_sys, time, float(price[0]),
                                 float(price[1]), float(price[2])))
        
    # 启动scheduler，程序每10秒钟执行一次查询和记录
    sched.add_job(log, 'interval', seconds=10)
    try:
        sched.start()
    except(KeyboardInterrupt):
        # 结束scheduler
        sched.shutdown(wait=False)


if __name__ == '__main__':
    
    main()
    input()
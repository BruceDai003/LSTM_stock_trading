# -*- coding: utf-8 -*-
# Copyright 2017 The Xiaoyu Fang. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
'''
Date: 2017/08/31
Author: Bruce Dai
Content: 修改extract_from_file及main函数里使用两层for循环逐个写入特征数据的方式，
         改为使用np.save()的方式，可以提高效率。
'''

import os
from time import time
from chart import extract_feature
import numpy as np
import pandas as pd


def timer_decorator(func):
    
    def wrapper(*args):
        t1 = time()
        func(*args)
        t2 = time()
        t_diff = t2 - t1
        print('-'*50)
        print('总耗时为%.2f秒' % t_diff)
        print('-'*50)
    return wrapper



def extract_all_features(data_dir, feature_dir, days_for_test, extract_all=True):
    '''
    从data_dir中依次读取原始数据，提取相应（训练集\测试集）特征，存入feature_dir。
    '''
    if extract_all == True:
        file_list = os.listdir(data_dir)
    else:
        file_list = ['%s.csv' % x for x in ['000016', '000300', '000905']]
        
    for i, f in enumerate(file_list):
        output_prefix = f.split(sep='.')[0]
        f = os.path.join(data_dir, f)
        extract_from_file(i, f, feature_dir, output_prefix, days_for_test)


def extract_from_file(idx, filepath, feature_dir, output_prefix, days_for_test):
    '''
    固定训练集的特征不变，无需再提取，只需要更新测试集的特征
    '''
    input_shape = [30, 61]  # [length of time series, length of feature]
    window = input_shape[0]
    #fp = os.path.join(feature_dir, "%s_feature.%s" % (output_prefix, window))
    #lp = os.path.join(feature_dir, "%s_label.%s" % (output_prefix, window))
    fpt = os.path.join(feature_dir,
                       "%s_feature.test.%s" % (output_prefix, window))
    lpt = os.path.join(feature_dir,
                       "%s_label.test.%s" % (output_prefix, window))

    selector = ["ROCP", "OROCP", "HROCP", "LROCP", "MACD", "RSI", "VROCP",
                "BOLL", "MA", "VMA", "PRICE_VOLUME"]

    raw_data = pd.read_csv(filepath, dtype={'volume': np.float})
    moving_features, moving_labels = extract_feature(raw_data=raw_data,
                                                     selector=selector,
                                                     window=input_shape[0],
                                                     with_label=True,
                                                     flatten=True)
    print("%02d\t%s特征提取完毕，开始写入文件..." % (idx+1, output_prefix))
    train_end_test_begin = moving_features.shape[0] - days_for_test
    #print('只更新测试集特征，一共%d个。' % days_for_test)
    if train_end_test_begin < 0:
        train_end_test_begin = 0
        print('data too short for %s' % output_prefix)
    #np.save(fp, moving_features[:train_end_test_begin, :])
    #np.save(lp, moving_labels[:train_end_test_begin])
    np.save(fpt, moving_features[train_end_test_begin:, :])
    np.save(lpt, moving_labels[train_end_test_begin:])
    print('写入完毕\n')
    print('-'*50)
    
@timer_decorator
def main():
    
    # 如果测试集截止到2017/08/31，默认测试最后759天
    days_for_test = 759
    input_shape = [30, 61]  # [length of time series, length of feature]
    window = input_shape[0]
    fp = "ultimate_feature.%s" % window
    lp = "ultimate_label.%s" % window
    fpt = "ultimate_feature.test.%s" % window
    lpt = "ultimate_label.test.%s" % window

    selector = ["ROCP", "OROCP", "HROCP", "LROCP", "MACD", "RSI", "VROCP",
                "BOLL", "MA", "VMA", "PRICE_VOLUME"]
    dataset_dir = "./dataset"
    for idx, filename in enumerate(os.listdir(dataset_dir)):
        #if filename != '000001.csv':
        #    continue
        print("%02d\t当前处理文件为：\t%s " % (idx, filename))
        filepath = os.path.join(dataset_dir, filename)
        raw_data = pd.read_csv(filepath, parse_dates=['date'])
        print('一共有%d个数据' % raw_data.shape[0])
        moving_features, moving_labels = extract_feature(raw_data=raw_data,
                                                         selector=selector,
                                                         window=input_shape[0],
                                                         with_label=True,
                                                         flatten=True)
        print("特征提取完毕")
        train_end_test_begin = moving_features.shape[0] - days_for_test

        if idx == 0:
            arr_fp = moving_features[:train_end_test_begin, :]
            arr_lp = moving_labels[:train_end_test_begin]
            arr_fpt = moving_features[train_end_test_begin:, :]
            arr_lpt = moving_labels[train_end_test_begin:]
        else:
            arr_fp = np.concatenate((arr_fp,
                                     moving_features[:train_end_test_begin, :]))
            arr_lp = np.concatenate((arr_lp,
                                     moving_labels[:train_end_test_begin]))
            arr_fpt = np.concatenate((arr_fpt,
                                      moving_features[train_end_test_begin:, :]))
            arr_lpt = np.concatenate((arr_lpt,
                                      moving_labels[train_end_test_begin:]))
        print('数据衔接完成...')
    print('-'*50)
    print('开始写入所有特征数据')
    np.save(fp, arr_fp)
    np.save(lp, arr_lp)
    np.save(fpt, arr_fpt)
    np.save(lpt, arr_lpt)        
    print('写入完毕\n')
    print('-' * 50)


if __name__ == '__main__':
    main()


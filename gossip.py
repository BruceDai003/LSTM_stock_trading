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
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import ffn
from keras.callbacks import TensorBoard, ModelCheckpoint

from windpuller import WindPuller
from dataset import DataSet
from feature import extract_from_file, extract_all_features


def read_ultimate(path, input_shape):
    ultimate_features = np.load(path + "ultimate_feature." + str(input_shape[0]) +'.npy')
    ultimate_features = np.reshape(ultimate_features, [-1, input_shape[0], input_shape[1]])
    ultimate_labels = np.load(path + "ultimate_label." + str(input_shape[0]) +'.npy')
    # ultimate_labels = np.reshape(ultimate_labels, [-1, 1])
    train_set = DataSet(ultimate_features, ultimate_labels)
    test_features = np.load(path + "ultimate_feature.test." + str(input_shape[0]) +'.npy')
    test_features = np.reshape(test_features, [-1, input_shape[0], input_shape[1]])
    test_labels = np.load(path + "ultimate_label.test." + str(input_shape[0]) +'.npy')
    # test_labels = np.reshape(test_labels, [-1, 1])
    test_set = DataSet(test_features, test_labels)
    return train_set, test_set


def read_feature(path, input_shape, prefix):
    ultimate_features = np.load("%s/%s_feature.%s.npy" % (path, prefix,
                                                         str(input_shape[0])))
    ultimate_features = np.reshape(ultimate_features,
                                   [-1, input_shape[0], input_shape[1]])
    ultimate_labels = np.load("%s/%s_label.%s.npy" % (path, prefix,
                                                     str(input_shape[0])))
    # ultimate_labels = np.reshape(ultimate_labels, [-1, 1])
    train_set = DataSet(ultimate_features, ultimate_labels)
    test_features = np.load("%s/%s_feature.test.%s.npy" %
                            (path, prefix, str(input_shape[0])))
    test_features = np.reshape(test_features,
                               [-1, input_shape[0], input_shape[1]])
    test_labels = np.load("%s/%s_label.test.%s.npy" % (path, prefix,
                                                       str(input_shape[0])))
    # test_labels = np.reshape(test_labels, [-1, 1])
    test_set = DataSet(test_features, test_labels)
    return train_set, test_set

def calculate_cumulative_return(labels, pred):
    cr = []
    if len(labels) <= 0:
        return cr
    cr.append(1. * (1. + labels[0] * pred[0]))
    for l in range(1, len(labels)):
        cr.append(cr[l-1] * (1 + labels[l] * pred[l]))
    cap = np.array(cr)
    cr = cap - 1
    return cr, cap

def calculate_cumulative_return_cost(labels, pred, fee=0.0002):
    '''计算累积收益率, 初始资金为1
    params:
        labels: 实际日收益率
        pred: 预测的每日仓位[0, 1]
    returns:
        cr: 累计收益率序列
        cap: 资金曲线
    '''
    n = len(labels)
    cap = np.ones(n+1)
    for i in range(1, n+1):
        cap[i] = cap[i-1] * (1 + labels[i-1] * pred[i-1] - np.abs(pred[i-1]) * fee)
    cr = cap - 1
    return cr, cap


def plot_returns(df, f, n, title, output_dir):
    '''画出资金曲线
    '''
    
    df.rename(columns={'Strategy':'策略', 
                       'Benchmark':'指数'}).plot(figsize=(24, 16))
    plt.xlabel('时间')
    plt.ylabel('累计收益率')    
    plt.title('%s %s_%s择时做多策略vs买入持有策略累计收益率' % (title, f, n),
              fontsize=22)
    
    fig_dir = os.path.join(output_dir, '资金曲线')
    if not(os.path.exists(fig_dir)):
        os.mkdir(fig_dir)
    fig_path = os.path.join(output_dir, '资金曲线',
                             '%s_%s' % (n, title))
    
    plt.savefig(fig_path)
    plt.close()
    print(' 资金曲线画图结束\n')
    print('-'*30)


def calc_perf(output, f, n, key, output_dir):
    '''统计各项表现，画出资金曲线，生成投资报告
    '''
    
    # 1. 数据预处理
    # df = output.set_index(keys='date')
    df = output.copy(deep=False)
    df.index = pd.to_datetime(df.index)
    df.drop(['Close', 'Pct_change', 'Position'], axis=1, inplace=True)
    df['Strategy'] = df['Cum_return'] + 1
    df['Benchmark'] = df['Buy_hold'] + 1
    df.drop(['Cum_return', 'Buy_hold'], axis=1, inplace=True)
    
    if key == 'train':
        title = '训练集'
    else:
        title = '测试集'
    # 2. 画出资金曲线
    plt.rcParams.update({'font.size': 18})
    plot_returns(df, f, n, title, output_dir)
    
    # 3. 画出策略和指数的相关矩阵图
    returns = df.to_returns().dropna()
    returns.rename(columns={'Strategy':'策略', 
                       'Benchmark':'指数'}).plot_corr_heatmap()
    plt.title('%s相关系数热度图' % n)
    
    cor_plt_dir = os.path.join(output_dir, '相关系数')
    if not(os.path.exists(cor_plt_dir)):
        os.mkdir(cor_plt_dir)
    cor_plt_path = os.path.join(cor_plt_dir,
                             '%s_%s' % (n, title))
    plt.savefig(cor_plt_path)
    plt.close()
    print('相关系数画图结束')
    print('-'*30)
    
    # 4. 计算策略表现
    perf = df.calc_stats()
    result = dict()
    result['天数'] = df.shape[0] - 1
    result['起始日期'] = perf['Strategy'].start.strftime('%Y-%m-%d')
    result['截至日期'] = perf['Strategy'].end.strftime('%Y-%m-%d')
    
    result['收益率'] = perf['Strategy'].total_return
    result['年化收益率'] = perf['Strategy'].cagr
    result['今年收益率'] = perf['Strategy'].ytd
    result['最近6个月收益率'] = perf['Strategy'].six_month
    result['最近3个月收益率'] = perf['Strategy'].three_month
    result['当月收益率'] = perf['Strategy'].mtd
    
    result['最大回撤'] = perf['Strategy'].max_drawdown
    details = perf['Strategy'].drawdown_details
    if details is None:
        result['最大回撤周期数'] = 0
        result['最长未创新高周期数'] = 0
        result['平均回撤周期数'] = 0
    else:
        result['最大回撤周期数'] = \
            int(details[details['drawdown'] == result['最大回撤']]['days'])
        result['最长未创新高周期数'] = \
            perf['Strategy'].drawdown_details['days'].max()
        result['平均回撤周期数'] = perf['Strategy'].avg_drawdown_days
    try:
        result['夏普比率'] = perf['Strategy'].daily_sharpe
    except ZeroDivisionError as e:
        print('夏普比率分母为0!')
        result['夏普比率'] = np.nan
        
    result['最好日收益率'] = perf['Strategy'].best_day
    result['最差日收益率'] = perf['Strategy'].worst_day
    result['最好月收益率'] = perf['Strategy'].best_month
    result['最差月收益率'] = perf['Strategy'].worst_month
    result['最好年收益率'] = perf['Strategy'].best_year
    result['最差年收益率'] = perf['Strategy'].worst_year
    
    if (output.Position != 0).sum() != 0:
        result['胜率'] = (output.Pct_change[output.Position != 0] > 0).sum() / (
                (output.Position != 0).sum())# 做多
        result['交易次数'] = (output.Position != 0).sum()
        result['满仓次数'] = (output.Position.abs() == 1).sum()
        result['平均仓位'] = np.abs(output.Position.mean())
        result['交易频率'] = result['天数'] / result['交易次数']
        result['满仓频率'] = result['天数'] / result['满仓次数']
    else:
        result['胜率'] = np.nan
        result['交易次数'] = 0
        result['满仓次数'] = 0
        result['平均仓位'] = 0
        result['交易频率'] = np.nan
        result['满仓频率'] = np.nan
        
    
    # 5. 将dict结果写入csv文件
    result_dir = os.path.join(output_dir, '投资报告')
    if not(os.path.exists(result_dir)):
        os.mkdir(result_dir)
    result_path = os.path.join(result_dir,
                               '%s_%s.csv' % (n, title))
    with open(result_path, 'w') as csv_file:
        csv.writer(csv_file).writerows(result.items())    
    


def evaluate_model(model_path, code, output_dir, input_shape=[30, 61]):
    extract_from_file("dataset/%s.csv" % code, output_dir, code)
    train_set, test_set = read_feature(output_dir, input_shape, code)
    saved_wp = WindPuller(input_shape).load_model(model_path)
    scores = saved_wp.evaluate(test_set.images, test_set.labels, verbose=0)
    print('Test loss:', scores[0])
    print('test accuracy:', scores[1])
    pred = saved_wp.predict(test_set.images, 1024)
    [cr, cap] = calculate_cumulative_return(test_set.labels, pred)

    # Output to a csv file
    # Read in the date, close from original data file.
    days_for_test = 700
    tmp = pd.read_csv('dataset/%s.csv' % code, delimiter='\t')
    # tmp.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    date = tmp['date'][-days_for_test:]
    close = tmp['close'][-days_for_test:]
    output = pd.DataFrame({'Return': test_set.labels,
                           'Position': pred.reshape(-1),
                           'Capital': cap.reshape(-1),
                           'Close': close.values},
                          index=date,
                          columns=['Close', 'Return', 'Position', 'Capital'])
    output.to_csv('output/%s.csv' % code)
    

def test_model(model_path="model.30.best", extract_all=True,
               days_for_test=False):
    '''
    1. 先对数据集中每一个品种提取特征；
    2. 读取训练集和验证集；
    3. 加载训练好的模型，预测在训练集和验证集上的结果；
    4. 根据结果绘制相应的资金变化图，并保存。
    '''
    
    # 1. 特征提取
    data_dir = './dataset/'
    output_dir = './output09/'
    feature_dir = './stock_features/'
    if not(os.path.exists(output_dir)):
        os.mkdir(output_dir)
    # 只提取测试集的特征
    if days_for_test == False:
        # 测试集从2017-09-01开始
        df = pd.read_csv('dataset/000001.csv', index_col='date',
                         parse_dates=True)
        days_for_test = df.shape[0] - df.index.get_loc('2017-09-01')
    
    extract_all_features(data_dir, feature_dir, days_for_test)
    
    # 2. 读取特征
    input_shape = [30, 61]
    file_list = os.listdir(data_dir)
    if extract_all == True:
        column_names = [s.split(sep='.')[0] for s in file_list]
    else:
        # 否则只测试3个指数
        column_names = ['000016', '000300', '000905']
        
    wp = WindPuller(input_shape).load_model(model_path)
    
    for f in column_names:
        
        train_set, test_set = read_feature(feature_dir, input_shape, f)
        data_set = {'train': train_set, 'test': test_set}
        tmp = pd.read_csv('dataset/%s.csv' % f)
        
        for key in data_set:
            # 3.分别给训练集/验证集预测并画图保存
            print('当前处理 %s_%s\n' % (f, key))
            val = data_set[key]
            pred = wp.predict(val.images, 1024)
            [cr, cap] = calculate_cumulative_return_cost(val.labels, pred)
            
            # 根据训练集/验证集来设置读取数据的范围
            if key == 'train':
                index = range(input_shape[0]-1, input_shape[0] + pred.shape[0])
            elif key == 'test':
                index = range(tmp.shape[0]-days_for_test-1, tmp.shape[0])
            
            # 1). 保存资金曲线的数据
            date = tmp['date'].iloc[index]
            close = tmp['close'].iloc[index]
            buy_hold = close / close.iloc[0] - 1
            # DEBUG:
            #print('date shape:\t', date.shape)
            #print('close shape:\t', close.shape)
            #print('buy_hold shape:\t', buy_hold.shape)
            #print('Pct_change shape:\t', val.labels.shape)
            #print('Position shape:\t', pred.shape)
            output = pd.DataFrame({'Close': close.values, 
                                   'Pct_change': np.concatenate(([np.nan],
                                                            val.labels)),
                                   'Position': 
                                       np.concatenate(([np.nan],
                                                      pred.reshape(-1))),
                                   'Cum_return': cr.reshape(-1),
                                   'Buy_hold': buy_hold.values},
                                  index=date,
                                  columns=['Close', 'Pct_change', 
                                           'Position', 'Cum_return',
                                           'Buy_hold'])
            names = pd.read_csv('指数名称.csv',
                              dtype={'code':np.str, 'name':np.str},
                              engine='python')
            names.set_index('code', inplace=True)
            names = names.to_dict()['name']
            n = names[f]
            
            # 写入文件
            cap_line_dir = os.path.join(output_dir, 'stocks')
            if not(os.path.exists(cap_line_dir)):
                os.mkdir(cap_line_dir)
            cap_line_f = os.path.join(cap_line_dir, '%s_%s.csv' % (n, key))
            output.to_csv(cap_line_f)
            
            # 2). 统计各项表现，画出资金曲线，生成投资报告
            print('开始计算策略表现 %s_%s_%s\n' % (f, n, key))
            calc_perf(output, f, n, key, output_dir)
            print('计算完毕')
            print('='*50)





def predict_tomorrow(model_path="model.30.best", extract_all=False):
    '''
    1. 先对3个数据集中每一个品种提取特征；
    2. 读取只有一行数据的验证集；
    3. 加载训练好的模型，预测在验证集上的信号结果；
    4. 保存信号结果。
    '''
    
    # 1. 特征提取
    data_dir = './newdata/'
    output_dir = './output09/'
    feature_dir = './stock_features/'
    if not(os.path.exists(output_dir)):
        os.mkdir(output_dir)
    
    # 测试集从2017-09-01开始
    df = pd.read_csv('dataset/000300.csv', index_col='date',
                     parse_dates=True)
    days_for_test = df.shape[0] - df.index.get_loc('2017-09-01')
    extract_all_features(data_dir, feature_dir, days_for_test, extract_all)
    
    # 2. 读取特征
    input_shape = [30, 61]
    file_list = os.listdir(data_dir)
    if extract_all == True:
        column_names = [s.split(sep='.')[0] for s in file_list]
    else:
        # 否则只测试3个指数
        column_names = ['000016', '000300', '000905']
    
    # 加载模型
    wp = WindPuller(input_shape).load_model(model_path)
    
    for f in column_names:
        
        _, test_set = read_feature(feature_dir, input_shape, f)
        tmp = pd.read_csv('dataset/%s.csv' % f)
        
        val = test_set
        pred = wp.predict(val.images, 1024)
        print(pred[-1])
        [cr, cap] = calculate_cumulative_return_cost(val.labels, pred)
        
        # 设置读取验证集数据的范围
        index = range(tmp.shape[0]-days_for_test-1, tmp.shape[0])
        
        # 1. 保存资金曲线的数据
        date = tmp['date'].iloc[index]
        close = tmp['close'].iloc[index]
        buy_hold = close / close.iloc[0] - 1
        output = pd.DataFrame({'Close': close.values, 
                               'Pct_change': np.concatenate(([np.nan],
                                                        val.labels)),
                               'Position': 
                                   np.concatenate(([np.nan],
                                                  pred.reshape(-1))),
                               'Cum_return': cr.reshape(-1),
                               'Buy_hold': buy_hold.values},
                              index=date,
                              columns=['Close', 'Pct_change', 
                                       'Position', 'Cum_return',
                                       'Buy_hold'])
        names = pd.read_csv('指数名称.csv',
                          dtype={'code':np.str, 'name':np.str},
                          engine='python')
        names.set_index('code', inplace=True)
        names = names.to_dict()['name']
        n = names[f]
        
        # 写入文件
        cap_line_dir = os.path.join(output_dir, 'stocks')
        if not(os.path.exists(cap_line_dir)):
            os.mkdir(cap_line_dir)
        cap_line_f = os.path.join(cap_line_dir, '%s_test.csv' % n)
        output.to_csv(cap_line_f)
        
        ## 2. 统计各项表现，画出资金曲线，生成投资报告
        #print('当前处理 %s_%s_test\n' % (f, n))
        #calc_perf(output, f, n, 'test', output_dir)
        print('计算完毕')
        print('='*50)


def make_model(input_shape, nb_epochs=100, batch_size=128, lr=0.01, n_layers=1, n_hidden=16, rate_dropout=0.3):
    model_path = 'model.%s' % input_shape[0]
    wp = WindPuller(input_shape=input_shape, lr=lr, n_layers=n_layers, n_hidden=n_hidden, rate_dropout=rate_dropout)
    train_set, test_set = read_ultimate("./", input_shape)
    wp.fit(train_set.images, train_set.labels, batch_size=batch_size,
           nb_epoch=nb_epochs, shuffle=True, verbose=1,
           validation_data=(test_set.images, test_set.labels),
           callbacks=[TensorBoard(histogram_freq=0, write_graph=0),
                      ModelCheckpoint(filepath=model_path+'.best', save_best_only=True, mode='min')])
    scores = wp.evaluate(test_set.images, test_set.labels, verbose=0)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    wp.model.save(model_path)
    saved_wp = wp.load_model(model_path)
    scores = saved_wp.evaluate(test_set.images, test_set.labels, verbose=0)
    print('Test loss:', scores[0])
    print('test accuracy:', scores[1])
    pred = saved_wp.predict(test_set.images, 1024)
    # print(pred)
    # print(test_set.labels)
    pred = np.reshape(pred, [-1])
    result = np.array([pred, test_set.labels]).transpose()
    with open('output.' + str(input_shape[0]), 'w') as fp:
        for i in range(result.shape[0]):
            for val in result[i]:
                fp.write(str(val) + "\t")
            fp.write('\n')

if __name__ == '__main__':
    operation = "train"
    if len(sys.argv) > 1:
        operation = sys.argv[1]
    if operation == "train":
        make_model([30, 61], 2000, 512, lr=0.001)
    elif operation == "predict":
        evaluate_model("model.30.best", '000001', 'stock_features', "000001")
    elif operation == 'test':
        test_model('model.30.best')
    else:
        print("Usage: gossip.py [train | predict]")

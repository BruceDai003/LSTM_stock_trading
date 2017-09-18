# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 10:50:32 2017

@author: brucedai
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ffn

from feature import timer_decorator


buy_dir = './output/stocks'
sell_dir = '../DeepTrade_Sell/output/stocks'
output_dir = './做多为主'    


def extract_capitals(data, direction='Buy'):
    '''对买入策略或卖出策略，进行数据的预处理，
    如丢弃某些列，转换净收益率为资金值等。
    '''
    df = data.copy(deep=False)
    df = df.set_index(keys='date')
    df.index = pd.to_datetime(df.index)
    df.drop(['Close', 'Pct_change', 'Position'], axis=1, inplace=True)
    df[direction] = df['Cum_return'] + 1
    df['Benchmark'] = df['Buy_hold'] + 1
    df.drop(['Cum_return', 'Buy_hold'], axis=1, inplace=True)
    return df


def plot_returns(df, f, n, title, output_dir):
    '''画出资金曲线
    '''
    
    df.rename(columns={'Buy':'买入策略',
                       'Sell': '卖出策略',
                       'Portfolio': '投资组合',
                       'Benchmark':'指数'}).plot(figsize=(24, 16))
    plt.xlabel('时间')
    plt.ylabel('资金')
    plt.title('%s %s_%s择时策略vs指数资金曲线' % (title, f, n),
              fontsize=22)
    plt.savefig(os.path.join(output_dir, '资金曲线',
                             '%s_%s' % (n, title)))
    plt.close()
    print('资金曲线画图结束\n')
    print('- -'*16)


def get_result_dict(df, perf, buy_df, sell_df):
    
    result = dict()
    keys = ['Buy', 'Sell', 'Benchmark', 'Portfolio']
    result['策略'] = ['买入策略', '卖出策略', '指数', '投资组合']
    result['天数'] = [df.shape[0] - 1] * df.shape[1]
    result['起始日期'] = [perf[k].start.strftime('%Y-%m-%d') for k in keys]
    result['截至日期'] = [perf[k].end.strftime('%Y-%m-%d') for k in keys]
    
    result['收益率'] = [perf[k].total_return for k in keys]
    result['年化收益率'] = [perf[k].cagr for k in keys]
    result['今年收益率'] = [perf[k].ytd for k in keys]
    result['最近6个月收益率'] = [perf[k].six_month for k in keys]
    result['最近3个月收益率'] = [perf[k].three_month for k in keys]
    result['当月收益率'] = [perf[k].mtd for k in keys]
    
    result['最大回撤'] = [perf[k].max_drawdown for k in keys]   
    details = [perf[k].drawdown_details for k in keys]
    result['最大回撤周期数'] = [int(d.loc[d['drawdown'] == p, 'days']) \
                               for (d, p) in zip(details, result['最大回撤'])]
    result['最长未创新高周期数'] = [d['days'].max() for d in details]
    
    result['平均回撤周期数'] = [perf[k].avg_drawdown_days for k in keys]
    
    result['夏普比率'] = [perf[k].daily_sharpe for k in keys]
    result['最好日收益率'] = [perf[k].best_day for k in keys]
    result['最差日收益率'] = [perf[k].worst_day for k in keys]
    result['最好月收益率'] = [perf[k].best_month for k in keys]
    result['最差月收益率'] = [perf[k].worst_month for k in keys]
    result['最好年收益率'] = [perf[k].best_year for k in keys]
    result['最差年收益率'] = [perf[k].worst_year for k in keys]
    
    result['胜率'] = [(buy_df.Pct_change[buy_df.Position != 0] > 0).sum() / (
            (buy_df.Position != 0).sum()),
            (sell_df.Pct_change[sell_df.Position != 0] < 0).sum() / (
            (sell_df.Position != 0).sum()),
            np.nan,
            ((buy_df.Pct_change[buy_df.Position != 0] > 0).sum() +
            (sell_df.Pct_change[sell_df.Position != 0] < 0).sum()) / (
            (buy_df.Position != 0).sum() + (sell_df.Position != 0).sum())]
    result['交易次数'] = [(buy_df.Position != 0).sum(),
          (sell_df.Position != 0).sum(), np.nan,
          (buy_df.Position != 0).sum() + (sell_df.Position != 0).sum()]
    result['满仓次数'] = [(buy_df.Position.abs() == 1).sum(),
          (sell_df.Position.abs() == 1).sum(), np.nan,
          (buy_df.Position.abs() == 1).sum() + \
          (sell_df.Position.abs() == 1).sum()]
    result['平均仓位'] = [np.abs(buy_df.Position.mean()),
          np.abs(sell_df.Position.mean()), np.nan,
          np.abs(buy_df.Position.mean()) + np.abs(sell_df.Position.mean())]
    result['交易频率'] = [t/j for (t, j) in \
                         zip(result['天数'], result['交易次数'])]
    result['满仓频率'] = [t/j for (t, j) in \
                         zip(result['天数'], result['满仓次数'])] 
    
    columns = ['策略', '天数', '起始日期', '截至日期', '收益率', '年化收益率',
               '今年收益率', '最近6个月收益率', '最近3个月收益率', '当月收益率',
               '最大回撤', '最大回撤周期数', '最长未创新高周期数', 
               '平均回撤周期数', '夏普比率', '最好日收益率', '最差日收益率',
               '最好月收益率', '最差月收益率', '最好年收益率', '最差年收益率',
               '胜率', '交易次数', '满仓次数', '平均仓位', '交易频率', '满仓频率']
    result = pd.DataFrame(result, columns=columns)
    result.set_index(keys=['策略'], inplace=True)
    result = result.T
    
    return result


def calc_perf(buy_df, sell_df, f, n, key, output_dir):
    '''统计各项表现，画出资金曲线，生成投资报告
    '''
    
    # 1. 合并buy_df和sell_df，保存每日仓位等信息
    # 设置相应标题
    if key == 'train':
        title = '训练集'
    else:
        title = '测试集'
    
    df_merge = pd.merge(buy_df, sell_df, on=['date', 'Close', 'Pct_change', 'Buy_hold'],
                        suffixes=('_Buy', '_Sell'))
    df_merge = df_merge.reindex(columns=['date', 'Close', 'Pct_change',
                                         'Position_Buy', 'Cum_return_Buy',
                                         'Position_Sell', 'Cum_return_Sell',
                                         'Buy_hold'])
    df_merge_path = os.path.join(output_dir, 'stocks',
                                 '%s_%s.csv' % (n, title))
    df_merge.to_csv(df_merge_path, index=False)
    
    # 2. 数据预处理
    # 买入数据
    df = extract_capitals(buy_df)
    # 卖出数据
    df2 = extract_capitals(sell_df, direction='Sell')
    # 合并买入卖出资金曲线
    df = df.merge(df2, left_index=True, right_index=True, on='Benchmark')
    df['Portfolio'] = (df['Buy'] + df['Sell'])/2
    df = df.reindex(columns=['Buy', 'Sell', 'Benchmark', 'Portfolio'])

    # 3. 画出资金曲线
    plt.rcParams.update({'font.size': 18})
    plot_returns(df, f, n, title, output_dir)
    
    # 4. 画出策略和指数的相关矩阵图
    returns = df.to_returns().dropna()
    returns.rename(columns={'Buy':'买入',
                       'Sell': '卖出',
                       'Portfolio': '组合',
                       'Benchmark':'指数'}).plot_corr_heatmap()
    plt.title('%s相关系数热度图' % n)
    plt.savefig(os.path.join(output_dir, '相关系数',
                             '%s_%s' % (n, title)))
    plt.close()
    print('相关系数画图结束')
    print('='*50)
    
    # 5. 计算策略表现
    perf = df.calc_stats()
    result = get_result_dict(df, perf, buy_df, sell_df)
    
    # 6. 保存计算结果
    result_path = os.path.join(output_dir, '投资报告',
                               '%s_%s.csv' % (n, title))    
    result.to_csv(result_path)


@timer_decorator
def main():
    
    names = pd.read_csv('指数名称.csv',
                      dtype={'code':np.str, 'name':np.str},
                      engine='python')
    names.set_index('code', inplace=True)
    names = names.to_dict()['name']
    
    train_test = ['train', 'test']
    for idx, f in enumerate(names):
        n = names[f]
        for key in train_test:
            print('%d\t当前处理 %s_%s_%s\n' % (idx+1, f, n, key))
            fname = '%s_%s.csv' % (n, key)
            buy_df = pd.read_csv(os.path.join(buy_dir, fname),
                                 parse_dates=['date'], engine='python')
            sell_df = pd.read_csv(os.path.join(sell_dir, fname),
                                  parse_dates=['date'], engine='python')
            calc_perf(buy_df, sell_df, f, n, key, output_dir)


if __name__ == '__main__':
    main()
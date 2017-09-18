# coding:UTF-8
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

import numpy
import talib


class ChartFeature(object):
    def __init__(self, selector):
        self.selector = selector
        self.supported = {"ROCP", "OROCP", "HROCP", "LROCP", "MACD", "RSI", "VROCP", "BOLL", "MA", "VMA", "PRICE_VOLUME"}
        self.feature = []

    def moving_extract(self, window=30, open_prices=None, close_prices=None, high_prices=None, low_prices=None,
                       volumes=None, with_label=True, flatten=True):
        self.extract(open_prices=open_prices, close_prices=close_prices, high_prices=high_prices, low_prices=low_prices,
                     volumes=volumes)
        feature_arr = numpy.asarray(self.feature)
        p = 0
        # rows = feature_arr.shape[0]
        # print("feature dimension: %s" % rows)
        if with_label:
            moving_features = []
            moving_labels = []
            while p + window <= feature_arr.shape[1]:
                x = feature_arr[:, p:p + window]
                # y = cmp(close_prices[p + window], close_prices[p + window - 1]) + 1
                if p + window < feature_arr.shape[1]:
                    p_change = (close_prices[p + window] - close_prices[p + window - 1]) / close_prices[p + window - 1]
                else:
                    p_change = 0
                # use percent of change as label
                y = p_change
                if flatten:
                    x = x.flatten("F")
                moving_features.append(numpy.nan_to_num(x))
                moving_labels.append(y)
                p += 1

            return numpy.asarray(moving_features), numpy.asarray(moving_labels)
        else:
            moving_features = []
            while p + window <= feature_arr.shape[1]:
                x = feature_arr[:, p:p + window]
                if flatten:
                    x = x.flatten("F")
                moving_features.append(numpy.nan_to_num(x))
                p += 1
            return moving_features

    def extract(self, open_prices=None, close_prices=None, high_prices=None, low_prices=None, volumes=None):
        self.feature = []
        for feature_type in self.selector:
            if feature_type in self.supported:
                # print("extracting feature : %s" % feature_type)
                self.extract_by_type(feature_type, open_prices=open_prices, close_prices=close_prices,
                                     high_prices=high_prices, low_prices=low_prices, volumes=volumes)
            else:
                print("feature type not supported: %s" % feature_type)
        # self.feature_distribution()
        return self.feature

    def feature_distribution(self):
        k = 0
        for feature_column in self.feature:
            fc = numpy.nan_to_num(feature_column)
            mean = numpy.mean(fc)
            var = numpy.var(fc)
            max_value = numpy.max(fc)
            min_value = numpy.min(fc)
            print("[%s_th feature] mean: %s, var: %s, max: %s, min: %s" % (k, mean, var, max_value, min_value))
            k = k + 1

    def extract_by_type(self, feature_type, open_prices=None, close_prices=None, high_prices=None, low_prices=None,
                        volumes=None):
        if feature_type == 'ROCP':
            rocp = talib.ROCP(close_prices, timeperiod=1)
            self.feature.append(rocp)
        if feature_type == 'OROCP':
            orocp = talib.ROCP(open_prices, timeperiod=1)
            self.feature.append(orocp)
        if feature_type == 'HROCP':
            hrocp = talib.ROCP(high_prices, timeperiod=1)
            self.feature.append(hrocp)
        if feature_type == 'LROCP':
            lrocp = talib.ROCP(low_prices, timeperiod=1)
            self.feature.append(lrocp)
        if feature_type == 'MACD':
            macd, signal, hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
            norm_signal = numpy.minimum(numpy.maximum(numpy.nan_to_num(signal), -1), 1)
            norm_hist = numpy.minimum(numpy.maximum(numpy.nan_to_num(hist), -1), 1)
            norm_macd = numpy.minimum(numpy.maximum(numpy.nan_to_num(macd), -1), 1)

            zero = numpy.asarray([0])
            macdrocp = numpy.minimum(numpy.maximum(numpy.concatenate((zero, numpy.diff(numpy.nan_to_num(macd)))), -1), 1)
            signalrocp = numpy.minimum(numpy.maximum(numpy.concatenate((zero, numpy.diff(numpy.nan_to_num(signal)))), -1), 1)
            histrocp = numpy.minimum(numpy.maximum(numpy.concatenate((zero, numpy.diff(numpy.nan_to_num(hist)))), -1), 1)

            self.feature.append(norm_macd)
            self.feature.append(norm_signal)
            self.feature.append(norm_hist)

            self.feature.append(macdrocp)
            self.feature.append(signalrocp)
            self.feature.append(histrocp)
        if feature_type == 'RSI':
            rsi6 = talib.RSI(close_prices, timeperiod=6)
            rsi12 = talib.RSI(close_prices, timeperiod=12)
            rsi24 = talib.RSI(close_prices, timeperiod=24)
            rsi6rocp = talib.ROCP(rsi6 + 100., timeperiod=1)
            rsi12rocp = talib.ROCP(rsi12 + 100., timeperiod=1)
            rsi24rocp = talib.ROCP(rsi24 + 100., timeperiod=1)
            self.feature.append(rsi6 / 100.0 - 0.5)
            self.feature.append(rsi12 / 100.0 - 0.5)
            self.feature.append(rsi24 / 100.0 - 0.5)
            # self.feature.append(numpy.maximum(rsi6 / 100.0 - 0.8, 0))
            # self.feature.append(numpy.maximum(rsi12 / 100.0 - 0.8, 0))
            # self.feature.append(numpy.maximum(rsi24 / 100.0 - 0.8, 0))
            # self.feature.append(numpy.minimum(rsi6 / 100.0 - 0.2, 0))
            # self.feature.append(numpy.minimum(rsi6 / 100.0 - 0.2, 0))
            # self.feature.append(numpy.minimum(rsi6 / 100.0 - 0.2, 0))
            # self.feature.append(numpy.maximum(numpy.minimum(rsi6 / 100.0 - 0.5, 0.3), -0.3))
            # self.feature.append(numpy.maximum(numpy.minimum(rsi6 / 100.0 - 0.5, 0.3), -0.3))
            # self.feature.append(numpy.maximum(numpy.minimum(rsi6 / 100.0 - 0.5, 0.3), -0.3))
            self.feature.append(rsi6rocp)
            self.feature.append(rsi12rocp)
            self.feature.append(rsi24rocp)
        if feature_type == 'VROCP':
            vrocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(numpy.maximum(volumes, 1), timeperiod=1)))
            # norm_volumes = (volumes - numpy.mean(volumes)) / math.sqrt(numpy.var(volumes))
            # vrocp = talib.ROCP(norm_volumes + numpy.max(norm_volumes) - numpy.min(norm_volumes), timeperiod=1)
            # self.feature.append(norm_volumes)
            self.feature.append(vrocp)
        if feature_type == 'BOLL':
            upperband, middleband, lowerband = talib.BBANDS(close_prices, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
            self.feature.append((upperband - close_prices) / close_prices)
            self.feature.append((middleband - close_prices) / close_prices)
            self.feature.append((lowerband - close_prices) / close_prices)
        if feature_type == 'MA':
            ma5 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=5))
            ma10 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=10))
            ma20 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=20))
            ma30 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=30))
            ma60 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=60))
            ma90 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=90))
            ma120 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=120))
            ma180 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=180))
            ma360 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=360))
            ma720 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=720))
            ma5rocp = talib.ROCP(ma5, timeperiod=1)
            ma10rocp = talib.ROCP(ma10, timeperiod=1)
            ma20rocp = talib.ROCP(ma20, timeperiod=1)
            ma30rocp = talib.ROCP(ma30, timeperiod=1)
            ma60rocp = talib.ROCP(ma60, timeperiod=1)
            ma90rocp = talib.ROCP(ma90, timeperiod=1)
            ma120rocp = talib.ROCP(ma120, timeperiod=1)
            ma180rocp = talib.ROCP(ma180, timeperiod=1)
            ma360rocp = talib.ROCP(ma360, timeperiod=1)
            ma720rocp = talib.ROCP(ma720, timeperiod=1)
            self.feature.append(ma5rocp)
            self.feature.append(ma10rocp)
            self.feature.append(ma20rocp)
            self.feature.append(ma30rocp)
            self.feature.append(ma60rocp)
            self.feature.append(ma90rocp)
            self.feature.append(ma120rocp)
            self.feature.append(ma180rocp)
            self.feature.append(ma360rocp)
            self.feature.append(ma720rocp)
            self.feature.append((ma5 - close_prices) / close_prices)
            self.feature.append((ma10 - close_prices) / close_prices)
            self.feature.append((ma20 - close_prices) / close_prices)
            self.feature.append((ma30 - close_prices) / close_prices)
            self.feature.append((ma60 - close_prices) / close_prices)
            self.feature.append((ma90 - close_prices) / close_prices)
            self.feature.append((ma120 - close_prices) / close_prices)
            self.feature.append((ma180 - close_prices) / close_prices)
            self.feature.append((ma360 - close_prices) / close_prices)
            self.feature.append((ma720 - close_prices) / close_prices)
        if feature_type == 'VMA':
            ma5 = talib.MA(volumes, timeperiod=5)
            ma10 = talib.MA(volumes, timeperiod=10)
            ma20 = talib.MA(volumes, timeperiod=20)
            ma30 = talib.MA(volumes, timeperiod=30)
            ma60 = talib.MA(volumes, timeperiod=60)
            ma90 = talib.MA(volumes, timeperiod=90)
            ma120 = talib.MA(volumes, timeperiod=120)
            ma180 = talib.MA(volumes, timeperiod=180)
            ma360 = talib.MA(volumes, timeperiod=360)
            ma720 = talib.MA(volumes, timeperiod=720)
            ma5rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma5, timeperiod=1)))
            ma10rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma10, timeperiod=1)))
            ma20rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma20, timeperiod=1)))
            ma30rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma30, timeperiod=1)))
            ma60rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma60, timeperiod=1)))
            ma90rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma90, timeperiod=1)))
            ma120rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma120, timeperiod=1)))
            ma180rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma180, timeperiod=1)))
            ma360rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma360, timeperiod=1)))
            ma720rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma720, timeperiod=1)))
            self.feature.append(ma5rocp)
            self.feature.append(ma10rocp)
            self.feature.append(ma20rocp)
            self.feature.append(ma30rocp)
            self.feature.append(ma60rocp)
            self.feature.append(ma90rocp)
            self.feature.append(ma120rocp)
            self.feature.append(ma180rocp)
            self.feature.append(ma360rocp)
            self.feature.append(ma720rocp)
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma5 - volumes) / (volumes + 1))))
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma10 - volumes) / (volumes + 1))))
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma20 - volumes) / (volumes + 1))))
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma30 - volumes) / (volumes + 1))))
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma60 - volumes) / (volumes + 1))))
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma90 - volumes) / (volumes + 1))))
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma120 - volumes) / (volumes + 1))))
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma180 - volumes) / (volumes + 1))))
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma360 - volumes) / (volumes + 1))))
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma720 - volumes) / (volumes + 1))))
        if feature_type == 'PRICE_VOLUME':
            rocp = talib.ROCP(close_prices, timeperiod=1)
            # norm_volumes = (volumes - numpy.mean(volumes)) / math.sqrt(numpy.var(volumes))
            # vrocp = talib.ROCP(norm_volumes + numpy.max(norm_volumes) - numpy.min(norm_volumes), timeperiod=1)
            vrocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(numpy.maximum(volumes, 1), timeperiod=1)))
            pv = rocp * vrocp
            self.feature.append(pv)


def extract_feature(raw_data, selector, window=30, with_label=True, flatten=True):
    chart_feature = ChartFeature(selector)
    closes = raw_data.close.values
    opens = raw_data.open.values
    highs = raw_data.high.values
    lows = raw_data.low.values
    volumes = raw_data.volume.values
    if with_label:
        moving_features, moving_labels = chart_feature.moving_extract(window=window, open_prices=opens,
                                                                      close_prices=closes,
                                                                      high_prices=highs, low_prices=lows,
                                                                      volumes=volumes, with_label=with_label,
                                                                      flatten=flatten)
        return moving_features, moving_labels
    else:
        moving_features = chart_feature.moving_extract(window=window, open_prices=opens, close_prices=closes,
                                                       high_prices=highs, low_prices=lows, volumes=volumes,
                                                       with_label=with_label, flatten=flatten)
        return moving_features


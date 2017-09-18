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


class RawData(object):
    def __init__(self, date, open, high, low, close, volume):
        self.date = date
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


def read_sample_data(path):
    f = path.split('.')[-2]
    print("-"*20, ' 读取股票%s ' % f, '-'*20)
    raw_data = []
    separator = "\t"
    with open(path, "r") as fp:
        for line in fp:
            if line.startswith("date"):  # ignore label line
                continue
            l = line[:-1]
            fields = l.split(separator)
            if len(fields) > 5:
                raw_data.append(RawData(fields[0], float(fields[1]), float(fields[2]), float(fields[3]), float(fields[4]), float(fields[5])))
    sorted_data = sorted(raw_data, key=lambda x: x.date)
    print("数据量\t%d." % len(sorted_data))
    return sorted_data

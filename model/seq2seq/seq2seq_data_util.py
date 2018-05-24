# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from utils.config import KddConfig
# For single varuable seq2seq model
# def generate_train_dev_set(ts, dev_set_proportion):
#     '''
#     args:
#         ts : pandas timeseries
#         dev_set_proportion : proportion of dev set in the KDD_CUP_2018 set.
#     '''
#     ts = ts.values
#     all_length = len(ts)
#     dev_length = int(dev_set_proportion * all_length)
#     dev = ts[-dev_length:]
#     train = ts[:-dev_length]

#     return train, dev

# def generate_training_data_for_seq2seq(ts, batch_size=10, input_seq_len=120, output_seq_len=48):
#     '''
#     Random generate training KDD_CUP_2018 with batch_size.
#     args:
#         ts : training time series to be used.
#         batch_size : batch_size for the training KDD_CUP_2018.
#         input_seq_len : length of input_seq to the encoder.
#         output_seq_len : length of output_seq of the decoder.
#     returns:
#         np.array(input_seq_x) shape : [batch_size, input_seq_len]
#         np.array(output_seq_y) shape : [batch_size, output_seq_len]
#     '''
#     # TS = np.array(ts)
#     TS = ts

#     total_start_points = len(TS) - input_seq_len - output_seq_len
#     start_x_idx = np.random.choice(range(total_start_points), batch_size)

#     input_seq = [TS[i:(i+input_seq_len)] for i in start_x_idx]
#     output_seq = [TS[(i+input_seq_len):(i+input_seq_len+output_seq_len)] for i in start_x_idx]

#     return np.array(input_seq), np.array(output_seq)

# def generate_dev_data_for_seq2seq(ts, input_seq_len=120, output_seq_len=48):

#     TS = ts
#     dev_set = []
#     total_start_points = len(TS) - input_seq_len - output_seq_len

#     for i in range(total_start_points):
#         input_seq = TS[i:(i+input_seq_len)]
#         output_seq = TS[(i+input_seq_len):(i+input_seq_len+output_seq_len)]
#         dev_set.append((input_seq, output_seq))

#     return dev_set


# def generate_x_y_data(ts, past_seq_length, future_sequence_length, batch_size):
#     """
#     Generate single feature KDD_CUP_2018 for seq2seq. Random choose batch_size KDD_CUP_2018.

#     args:
#         ts is single feature time series. ts can be training KDD_CUP_2018 or validation KDD_CUP_2018 or test KDD_CUP_2018.
#         past_seq_length is seq_length of past KDD_CUP_2018.
#         future_sequence_length is sequence_length of future KDD_CUP_2018.
#         batch_size.

#     returns: tuple (X, Y)
#         X is (past_seq_length, batch_size, input_dim)
#         Y is (future_sequence_length, batch_size, output_dim)

#     """
#     series = ts.values

#     batch_x = []
#     batch_y = []

#     for _ in range(batch_size):

#         total_series_num = len(series) - (past_seq_length + future_sequence_length)
#         random_index = int(np.random.choice(total_series_num, 1))

#         x_ = series[random_index : random_index + past_seq_length]
#         y_ = series[random_index + past_seq_length : random_index + past_seq_length + future_sequence_length]


#         batch_x.append(x_)
#         batch_y.append(y_)

#     batch_x = np.array(batch_x)
#     batch_y = np.array(batch_y)

#     batch_x = np.expand_dims(batch_x, axis=2)
#     batch_y = np.expand_dims(batch_y, axis=2)
#     # shape: (batch_size, seq_length, input/output_dim)

#     batch_x = np.array(batch_x).transpose((1, 0, 2))
#     batch_y = np.array(batch_y).transpose((1, 0, 2))
#     # shape: (seq_length, batch_size, input/output_dim)

#     return batch_x, batch_y


# For multi variable seq2seq model

# def generate_train_samples(x, y, batch_size=32, input_seq_len=30, output_seq_len=5):

#     total_start_points = len(x) - input_seq_len - output_seq_len
#     start_x_idx = np.random.choice(range(total_start_points), batch_size, replace = False)

#     input_batch_idxs = [list(range(i, i+input_seq_len)) for i in start_x_idx]
#     input_seq = np.take(x, input_batch_idxs, axis = 0)

#     output_batch_idxs = [list(range(i+input_seq_len, i+input_seq_len+output_seq_len)) for i in start_x_idx]
#     output_seq = np.take(y, output_batch_idxs, axis = 0)

#     return input_seq, output_seq # in shape: (batch_size, time_steps, feature_dim)

# def generate_test_samples(x, y, input_seq_len=30, output_seq_len=5):

#     total_samples = x.shape[0]

#     input_batch_idxs = [list(range(i, i+input_seq_len)) for i in range((total_samples-input_seq_len-output_seq_len))]
#     input_seq = np.take(x, input_batch_idxs, axis = 0)

#     output_batch_idxs = [list(range(i+input_seq_len, i+input_seq_len+output_seq_len)) for i in range((total_samples-input_seq_len-output_seq_len))]
#     output_seq = np.take(y, output_batch_idxs, axis = 0)

#     return input_seq, output_seq

processed_data_dir = KddConfig.processed_data_dir
# test_data_dir="test_new/"


def get_training_statistics(city, eval=False):
    '''
    Get statics values of aq and meo KDD_CUP_2018.
    '''
    processed_data_dir = ''
    if eval:
        processed_data_dir = KddConfig.eval_processed_data_dir
    else:
        processed_data_dir = KddConfig.processed_data_dir

    aq_describe = pd.read_csv(processed_data_dir+"%s_aq_describe.csv" % (city))
    aq_describe.set_index("Unnamed: 0", inplace=True)

    meo_describe = pd.read_csv(
        processed_data_dir+"%s_meo_describe.csv" % (city))
    meo_describe.set_index("Unnamed: 0", inplace=True)

    statistics = pd.concat([aq_describe, meo_describe], axis=1)
    statistics = statistics.loc[["mean", "std"]]
    return statistics


# 可以按照 站点名称，特征名称，数据天数来灵活的生成验证集

def generate_training_set(city="bj", station_list=None, X_aq_list=None, y_aq_list=None, X_meo_list=None, use_day=True, pre_days=5, batch_size=32, gap=0):
    '''

    Args:
        station_list : a list of used stations.
        X_aq_list : a list of used aq features as input.
        y_aq_list : a list of used aq features as output. 
        X_meo_list : a list of used meo features.
        use_day : bool, True to just use 0-24 h days.
        pre_days : use pre_days history days to predict.
        batch_size : batch_size.
        gap : 0 or 12 or 24. 
                0 : 当天 23点以后进行的模型训练
                12 : 当天中午进行的模型训练
                24 : 不使用当天数据进行的训练

    station_list = ['dongsi_aq','tiantan_aq','guanyuan_aq','wanshouxigong_aq','aotizhongxin_aq',
                'nongzhanguan_aq','wanliu_aq','beibuxinqu_aq','zhiwuyuan_aq','fengtaihuayuan_aq',
                'yungang_aq','gucheng_aq','fangshan_aq','daxing_aq','yizhuang_aq','tongzhou_aq',
                'shunyi_aq','pingchang_aq','mentougou_aq','pinggu_aq','huairou_aq','miyun_aq',
                'yanqin_aq','dingling_aq','badaling_aq','miyunshuiku_aq','donggaocun_aq',
                'yongledian_aq','yufa_aq','liulihe_aq','qianmen_aq','yongdingmennei_aq',
                'xizhimenbei_aq','nansanhuan_aq','dongsihuan_aq']            
    X_aq_list = ["PM2.5","PM10","O3","CO","SO2","NO2"]  
    y_aq_list = ["PM2.5","PM10","O3"]
    X_meo_list = ["temperature","pressure","humidity","direction","speed/kph"]

    ld_station_list = ['BL0','CD1','CD9','GN0','GN3','GR4','GR9','HV1','KF1','LW2','MY7','ST5','TH4']            
    ld_X_aq_list = ['NO2', 'PM10', 'PM2.5']  
    ld_y_aq_list = ['PM10', 'PM2.5'] 
    ld_X_meo_list = ["temperature","pressure","humidity","direction","speed"]

    '''

    aq_train = pd.read_csv(processed_data_dir+"%s_aq_train_data.csv" % (city))
    meo_train = pd.read_csv(
        processed_data_dir+"%s_meo_train_data.csv" % (city))
    # print("shape of aq and meo training KDD_CUP_2018 are : ", aq_train.shape, meo_train.shape)

    train_df = pd.concat([aq_train, meo_train], axis=1)
    # train_df.to_csv('train_data.csv')
    # print 'aq_train.shape:', aq_train.shape, 'meo_train:', meo_train.shape   # aq_train.shape: (10873, 211) meo_train: (0, 176)
    # print 'train_df.shape:', train_df.shape                                  # train_df.shape: (10873, 387)

    '''
    for index in train_df.index:
        row = train_df.loc[index]
        if pd.isnull(row).any() :
            print(row)
            print 'index:', 
    '''
    # 统计数据有多少条
    original = train_df.shape[0]
    train_df = train_df.dropna()
    train_df.reset_index(inplace=True)
    drop_dim = train_df.shape[0]
    # print 'nan is %d' % (original-drop_dim)

    # print('ld X_feature:', train_df.shape)  out: (10873, 161)
    # print list(train_df)[:105]

    # print train_df.head()
    '''
    # print('aq_train column:', aq_train.shape[1], 'meo_train column:', aq_train.shape[1], 'train_df column:', train_df.shape[1])
    # ('aq_train column:', 211, 'meo_train column:', 176, 'train_df column:', 387)
    set_1 = set(list(aq_train))
    set_2 = set(list(meo_train))
    set_3 = set(list(train_df))
    print len(set_1), len(set_2), len(set_1 | set_2), len(set_3)
    cro = set_1 & set_2
    sub = (set_1 | set_2) - set_3
    print cro, sub
    '''
    # step 1 : keep all features about the stations
    # print 'before x_feature filter:', len(train_df.columns) out: 161
    station_filters = []
    for station in station_list:
        station_filter = [
            index for index in train_df.columns if station in index]
        station_filters += station_filter
    # print 'after filter , train station_filters:', station_filters, len(station_filters)
    '''
    104
    '''
    # print('step 1 successful!')
    # step 2 : filter of X features
    X_feature_filters = []
    if X_meo_list:
        X_features = X_aq_list + X_meo_list
    else:
        X_features = X_aq_list

    # print 'X_features:', X_features
    # out: ['NO2', 'PM10', 'PM2.5', 'temperature', 'pressure', 'humidity', 'direction', 'speed']
    # print 'station_filters:', station_filters out: 104 个特征
    if city == 'ld':
        X_feature_filters = station_filters
    else:
        for i in station_filters:
            if i.split("_")[-1] in X_features:
                X_feature_filters += [i]

    X_feature_filters.sort()  # 排序，保证训练集和验证集中的特征的顺序一致
    # print 'train X_feature_filters:', X_feature_filters

    X_df = train_df[X_feature_filters]
    # X_df.to_csv('./X_d.csv')
    # print(X_df.columns)
    # print('step 2 successful!')

    # step 3 : filter of y features
    y_feature_filters = []
    y_features = y_aq_list

    if city == 'ld':
        for i in station_filters:
            for j in y_features:
                if j in i:
                    y_feature_filters += [i]
    else:
        for i in station_filters:
            if i.split("_")[-1] in y_features:
                y_feature_filters += [i]

    y_feature_filters.sort()  # 排序，保证训练集和验证集中的特征的顺序一致
    # print 'train y_feature:', y_feature_filters, len(y_feature_filters)
    y_df = train_df[y_feature_filters]
    # print(y_df.columns)
    # print('step 3 successful!')

    # step 4 : generate training batch
    X_df_list = []
    y_df_list = []

    max_start_points = X_df.shape[0] - (pre_days + 2) * 24 - gap
    # print('max_start_points:', max_start_points)
    if use_day:
        total_start_points = range(0, max_start_points, 24)
    else:
        total_start_points = range(0, max_start_points, 1)
    # print('ok')
    for i in range(batch_size):
        # print(i)
        flag = True
        while flag:
            # print('excute while!')
            X_start_index = int(np.random.choice(
                total_start_points, 1, replace=False))
            X_end_index = X_start_index + pre_days * 24 - 1 - gap

            y_start_index = X_start_index + pre_days * 24
            y_end_index = y_start_index + 47

            # print(X_start_index, X_end_index, y_start_index, y_end_index)
            # print ('delta:', X_end_index-X_start_index)
            X = X_df.loc[X_start_index: X_end_index]
            # X = X_df.loc[X_start_index: X_start_index+119]
            # print('X.shape:', X.shape)
            y = y_df.loc[y_start_index: y_end_index]
            # X[X.isnull().values == True]
            # print pd.isnull(X)
            # 判断是不是有 NAN
            if pd.isnull(X).any().any() or pd.isnull(y).any().any():
                # print('exits Nall')
                # assert False
                pass
            else:
                X = np.array(X)
                # print('X.shape:', X.shape)
                y = np.array(y)
                X = np.expand_dims(X, axis=0)
                y = np.expand_dims(y, axis=0)
                # print('X.shape:', X.shape)
                X_df_list.append(X)
                y_df_list.append(y)
                flag = False
    # print('step 4 successful!')
    '''
    np.concatenate: Join a sequence of arrays along an existing axis.
    '''
    X_train_batch = np.concatenate(X_df_list, axis=0)
    y_train_batch = np.concatenate(y_df_list, axis=0)

    return X_train_batch, y_train_batch


def generate_X_test_set(city="bj",
                        station_list=None,
                        X_aq_list=None,
                        X_meo_list=None,
                        pre_days=5,
                        gap=0):

    aq_dev = pd.read_csv(processed_data_dir+"%s_aq_norm_data.csv" % (city))
    meo_dev = pd.read_csv(processed_data_dir+"%s_meo_norm_data.csv" % (city))

    # print('aq_dev.shape, meo_dev.shape:', aq_dev.shape, meo_dev.shape)
    dev_df = pd.concat([aq_dev, meo_dev], axis=1)
    # print 'dev_df.shape:', dev_df
    # 以下代码为改动代码
    # int len(dev_df)
    # dev_df.to_csv('havealookat.csv')
    dev_df = dev_df.dropna(axis=0)
    # dev_df.reset_index(inplace=True)
    # dev_df.to_csv('havealookat2.csv')
    # print len(dev_df)
    # out:
    #    1080
    #    683
    # 数据预处理部分不纯洁，缺失值比较严重
    # 改动结束
    # print('A'*10)
    # print(dev_df.shape)

    # step 1 : keep all features about the stations
    station_filters = []
    for station in station_list:
        station_filter = [
            index for index in dev_df.columns if station in index]
        station_filters += station_filter

    # step 2 : filter of X features
    X_feature_filters = []
    if X_meo_list:
        X_features = X_aq_list + X_meo_list
    else:
        X_features = X_aq_list

    if city == 'ld':
        X_feature_filters = station_filters
    else:
        for i in station_filters:
            if i.split("_")[-1] in X_features:
                X_feature_filters += [i]

    X_feature_filters.sort()  # 排序，保证训练集和验证集中的特征的顺序一致
    # print 'test X_feature_filters:', X_feature_filters

    X_df = dev_df[X_feature_filters]

    # step 3 : 根据 pre_days 和　gap，确定　test　中Ｘ的值
    delta = 0
    X_end_index = X_df.shape[0] - 1 - delta
    # X_end_index = X_df.shape[0] - 1 - delta

    X_start_index = X_end_index - pre_days * 24 + gap + 1

    X = X_df.loc[X_start_index: X_end_index]
    X = np.array(X)
    X = np.expand_dims(X, axis=0)

    return X


def generate_dev_set(city="bj", station_list=None, X_aq_list=None, y_aq_list=None, X_meo_list=None, pre_days=5, gap=0):
    '''

    Args:
        station_list : a list of used stations.
        X_aq_list : a list of used aq features as input.
        y_aq_list : a list of used aq features as output. 
        X_meo_list : a list of used meo features.

    station_list = ['dongsi_aq','tiantan_aq','guanyuan_aq','wanshouxigong_aq','aotizhongxin_aq',
                'nongzhanguan_aq','wanliu_aq','beibuxinqu_aq','zhiwuyuan_aq','fengtaihuayuan_aq',
                'yungang_aq','gucheng_aq','fangshan_aq','daxing_aq','yizhuang_aq','tongzhou_aq',
                'shunyi_aq','pingchang_aq','mentougou_aq','pinggu_aq','huairou_aq','miyun_aq',
                'yanqin_aq','dingling_aq','badaling_aq','miyunshuiku_aq','donggaocun_aq',
                'yongledian_aq','yufa_aq','liulihe_aq','qianmen_aq','yongdingmennei_aq',
                'xizhimenbei_aq','nansanhuan_aq','dongsihuan_aq']            
    X_aq_list = ["PM2.5","PM10","O3","CO","SO2","NO2"]  
    y_aq_list = ["PM2.5","PM10","O3"]
    X_meo_list = ["temperature","pressure","humidity","direction","speed/kph"]

    '''
    aq_dev = pd.read_csv(processed_data_dir+"%s_aq_dev_data.csv" % (city))
    meo_dev = pd.read_csv(processed_data_dir + "%s_meo_dev_data.csv" % (city))

    # print(aq_dev.shape)
    # print(meo_dev.shape)

    dev_df = pd.concat([aq_dev, meo_dev], axis=1)

    # step 1 : keep all features about the stations
    station_filters = []
    for station in station_list:
        station_filter = [
            index for index in dev_df.columns if station in index]
        station_filters += station_filter

    # step 2 : filter of X features
    X_feature_filters = []
    if X_meo_list:
        X_features = X_aq_list + X_meo_list
    else:
        X_features = X_aq_list
    if city == 'ld':
        X_feature_filters = station_filters
    else:
        for i in station_filters:
            if i.split("_")[-1] in X_features:
                X_feature_filters += [i]

    X_feature_filters.sort()  # 排序，保证训练集和验证集中的特征的顺序一致
    # print(len(X_feature_filters))
    X_df = dev_df[X_feature_filters]
    # print(X_df.columns)

    # step 3 : filter of y features
    y_feature_filters = []
    y_features = y_aq_list

    if city == 'ld':
        for i in station_filters:
            for j in y_features:
                if j in i:
                    y_feature_filters += [i]
    else:
        for i in station_filters:
            if i.split("_")[-1] in y_features:
                y_feature_filters += [i]

    y_feature_filters.sort()  # 排序，保证训练集和验证集中的特征的顺序一致
    y_df = dev_df[y_feature_filters]
    # print(y_df.columns)

    # step 4 : 按天生成数据
    X_df_list = []
    y_df_list = []

    min_y_start_index = 7 * 24  # 7 是当前的最长的 pre_days
    # max_y_start_index = X_df.shape[0] - 2 * 24
    max_y_start_index = X_df.shape[0] - 2 * 24

    for y_start_index in range(min_y_start_index, max_y_start_index, 24):

        X_start_index = y_start_index - pre_days * 24
        X_end_index = y_start_index - 1 - gap

        y_end_index = y_start_index + 47

        X = X_df.loc[X_start_index: X_end_index]
        y = y_df.loc[y_start_index: y_end_index]

        X = np.array(X)
        y = np.array(y)

        X = np.expand_dims(X, axis=0)
        y = np.expand_dims(y, axis=0)

        X_df_list.append(X)
        y_df_list.append(y)
    # print("AAAA")
    # print(len(X_df_list))

    X_dev_batch = np.concatenate(X_df_list, axis=0)
    y_dev_batch = np.concatenate(y_df_list, axis=0)

    return X_dev_batch, y_dev_batch


def generate_eval_set(city="bj", station_list=None, X_aq_list=None, y_aq_list=None, X_meo_list=None, pre_days=5, gap=0, start_day=0):
    '''

    Args:
        station_list : a list of used stations.
        X_aq_list : a list of used aq features as input.
        y_aq_list : a list of used aq features as output.
        X_meo_list : a list of used meo features.

    station_list = ['dongsi_aq','tiantan_aq','guanyuan_aq','wanshouxigong_aq','aotizhongxin_aq',
                'nongzhanguan_aq','wanliu_aq','beibuxinqu_aq','zhiwuyuan_aq','fengtaihuayuan_aq',
                'yungang_aq','gucheng_aq','fangshan_aq','daxing_aq','yizhuang_aq','tongzhou_aq',
                'shunyi_aq','pingchang_aq','mentougou_aq','pinggu_aq','huairou_aq','miyun_aq',
                'yanqin_aq','dingling_aq','badaling_aq','miyunshuiku_aq','donggaocun_aq',
                'yongledian_aq','yufa_aq','liulihe_aq','qianmen_aq','yongdingmennei_aq',
                'xizhimenbei_aq','nansanhuan_aq','dongsihuan_aq']
    X_aq_list = ["PM2.5","PM10","O3","CO","SO2","NO2"]
    y_aq_list = ["PM2.5","PM10","O3"]
    X_meo_list = ["temperature","pressure","humidity","direction","speed/kph"]

    '''

    aq_dev = pd.read_csv(KddConfig.eval_processed_data_dir +
                         "%s_aq_norm_data.csv" % (city))
    meo_dev = pd.read_csv(
        KddConfig.eval_processed_data_dir + "%s_meo_norm_data.csv" % (city))

    print('city:', city, aq_dev.shape)
    # print('aq_dev.shape, meo_dev.shape:', aq_dev.shape, meo_dev.shape)
    dev_df = pd.concat([aq_dev, meo_dev], axis=1)
    # print 'dev_df.shape:', dev_df
    # 以下代码为改动代码
    before_drop = len(dev_df)
    # dev_df.to_csv('havealookat.csv')
    dev_df = dev_df.dropna(axis=0)
    after_drop = len(dev_df)

    print('eval empty values number:', before_drop-after_drop)
    # dev_df.reset_index(inplace=True)
    # dev_df.to_csv('havealookat2.csv')
    # print len(dev_df)
    # out:
    #    1080
    #    683
    # 数据预处理部分不纯洁，缺失值比较严重
    # 改动结束
    # print('A'*10)
    # print(dev_df.shape)

    # step 1 : keep all features about the stations
    station_filters = []
    for station in station_list:
        station_filter = [
            index for index in dev_df.columns if station in index]
        station_filters += station_filter

    # step 2 : filter of X features
    X_feature_filters = []
    if X_meo_list:
        X_features = X_aq_list + X_meo_list
    else:
        X_features = X_aq_list

    if city == 'ld':
        X_feature_filters = station_filters
    else:
        for i in station_filters:
            if i.split("_")[-1] in X_features:
                X_feature_filters += [i]

    X_feature_filters.sort()  # 排序，保证训练集和验证集中的特征的顺序一致
    # print 'test X_feature_filters:', X_feature_filters

    X_df = dev_df[X_feature_filters]

    y_feature_filters = []
    y_features = y_aq_list

    if city == 'ld':
        for i in station_filters:
            for j in y_features:
                if j in i:
                    y_feature_filters += [i]
    else:
        for i in station_filters:
            if i.split("_")[-1] in y_features:
                y_feature_filters += [i]

    y_feature_filters.sort()  # 排序，保证训练集和验证集中的特征的顺序一致
    y_df = dev_df[y_feature_filters]

    # step 3 : 根据 pre_days 和　gap，确定　test　中Ｘ的值
    delta = 0
    # X_end_index = X_df.shape[0] - 1 - delta - 24 - 24*start_day
    X_end_index = X_df.shape[0] - 1 - delta - 48 - 24 * start_day
    Y_end_index = X_end_index + 48

    X_start_index = X_end_index - pre_days * 24 + gap + 1
    Y_start_index = Y_end_index - 47

    X = X_df.loc[X_start_index: X_end_index]
    Y = y_df.loc[Y_start_index: Y_end_index]
    X = np.array(X)
    Y = np.array(Y)

    X = np.expand_dims(X, axis=0)
    Y = np.expand_dims(Y, axis=0)

    print(X.shape, Y.shape)

    return X, Y


if __name__ == '__main__':
    bj_station_list = ['dongsi_aq', 'tiantan_aq', 'guanyuan_aq', 'wanshouxigong_aq', 'aotizhongxin_aq',
                       'nongzhanguan_aq', 'wanliu_aq', 'beibuxinqu_aq', 'zhiwuyuan_aq', 'fengtaihuayuan_aq',
                       'yungang_aq', 'gucheng_aq', 'fangshan_aq', 'daxing_aq', 'yizhuang_aq', 'tongzhou_aq',
                       'shunyi_aq', 'pingchang_aq', 'mentougou_aq', 'pinggu_aq', 'huairou_aq', 'miyun_aq',
                       'yanqin_aq', 'dingling_aq', 'badaling_aq', 'miyunshuiku_aq', 'donggaocun_aq',
                       'yongledian_aq', 'yufa_aq', 'liulihe_aq', 'qianmen_aq', 'yongdingmennei_aq',
                       'xizhimenbei_aq', 'nansanhuan_aq', 'dongsihuan_aq']
    bj_X_aq_list = ["PM2.5", "PM10", "O3", "CO", "SO2", "NO2"]
    bj_y_aq_list = ["PM2.5", "PM10", "O3"]
    bj_X_meo_list = ["temperature", "pressure",
                     "humidity", "direction", "speed/kph"]

    ld_station_list = ['BL0', 'CD1', 'CD9', 'GN0', 'GN3',
                       'GR4', 'GR9', 'HV1', 'KF1', 'LW2', 'MY7', 'ST5', 'TH4']
    ld_X_aq_list = ['NO2', 'PM10', 'PM2.5']
    ld_y_aq_list = ['PM10', 'PM2.5']
    ld_X_meo_list = ["temperature", "pressure",
                     "humidity", "direction", "speed"]

    #  batch_input, batch_output = generate_eval_set(city='ld',
    #                                                   station_list=ld_station_list,
    #                                                   X_aq_list=ld_X_aq_list, y_aq_list=ld_y_aq_list,
    #                                                   X_meo_list=ld_X_meo_list,
    #                                                   pre_days=5,
    #                                                   gap=0,
    #                                                    start_day=0,
    #                                                    )
    # 0.333
    batch_input, batch_output = generate_eval_set(city='bj',
                                                  station_list=bj_station_list,
                                                  X_aq_list=bj_X_aq_list, y_aq_list=bj_y_aq_list,
                                                  X_meo_list=bj_X_meo_list,
                                                  pre_days=5,
                                                  gap=0,
                                                       start_day=0,
                                                  )

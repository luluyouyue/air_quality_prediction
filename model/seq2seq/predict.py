# -*- coding: utf-8 -*-
#  训练一个使用所有特征的模型
# import tkinter
import os
import sys
import pandas as pd
import numpy as np
# import seaborn as sns
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

sys.path.append('../../metrics')

from utils.plot_util import plot_forecast_and_actual_example
from metrics import SMAPE_on_dataset_v1
from seq2seq_data_util import get_training_statistics, generate_training_set, generate_dev_set, generate_X_test_set
# from multi_variable_seq2seq_model_parameters import build_graph
from seq2seq_model import build_graph
from generate_submission import generator_result

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
session = tf.Session(config=gpu_config)
KTF.set_session(session)

# Args
bj_station_list = ['dongsi_aq', 'tiantan_aq', 'guanyuan_aq', 'wanshouxigong_aq', 'aotizhongxin_aq',
                   'nongzhanguan_aq', 'wanliu_aq', 'beibuxinqu_aq', 'zhiwuyuan_aq', 'fengtaihuayuan_aq',
                   'yungang_aq', 'gucheng_aq', 'fangshan_aq', 'daxing_aq', 'yizhuang_aq', 'tongzhou_aq',
                   'shunyi_aq', 'pingchang_aq', 'mentougou_aq', 'pinggu_aq', 'huairou_aq', 'miyun_aq',
                   'yanqin_aq', 'dingling_aq', 'badaling_aq', 'miyunshuiku_aq', 'donggaocun_aq',
                   'yongledian_aq', 'yufa_aq', 'liulihe_aq', 'qianmen_aq', 'yongdingmennei_aq',
                   'xizhimenbei_aq', 'nansanhuan_aq', 'dongsihuan_aq']
bj_X_aq_list = ["PM2.5", "PM10", "O3", "CO", "SO2", "NO2"]
bj_y_aq_list = ["PM2.5", "PM10", "O3"]
bj_X_meo_list = ["temperature", "pressure", "humidity", "direction", "speed"]

ld_station_list = ['BL0', 'CD1', 'CD9', 'GN0', 'GN3', 'GR4', 'GR9', 'HV1', 'KF1', 'LW2', 'MY7', 'ST5', 'TH4']
ld_X_aq_list = ['NO2', 'PM10', 'PM2.5']
ld_y_aq_list = ['PM10', 'PM2.5']
ld_X_meo_list = ["temperature", "pressure", "humidity", "direction", "speed"]  # "wind_direction","wind_speed"


# def train_and_dev(city='bj', pre_days=5, gap=0, loss_function="L2", total_iteractions=200):

def train_and_dev(city='ld', pre_days=5, gap=0, loss_function="L2", model_name=''):
    '''
    city='bj' or 'ld' : 针对某个城市的数据进行训练
    pre_days : 使用 pre_days 天数的数据进行预测
    gap : 0,12,24
        0 : 当天 23点以后进行的模型训练
        12 : 当天中午进行的模型训练
        24 : 不使用当天数据进行的训练
    loss_function : 使用不同的损失函数
    '''
    if city == "bj":
        station_list = bj_station_list
        X_aq_list = bj_X_aq_list
        y_aq_list = bj_y_aq_list
        X_meo_list = bj_X_meo_list
        model_path = './result_2/0430/'
    elif city == "ld":
        station_list = ld_station_list
        X_aq_list = ld_X_aq_list
        y_aq_list = ld_y_aq_list
        X_meo_list = ld_X_meo_list
        model_path = './result_2/ld/'

    use_day = True
    learning_rate = 1e-3
    batch_size = 128
    input_seq_len = pre_days * 24 - gap
    output_seq_len = 48
    hidden_dim = 256
    input_dim = len(station_list) * (len(X_aq_list) + len(X_meo_list))
    # input_dim = 350
    output_dim = len(station_list) * len(y_aq_list)
    # print(input_dim, output_dim) out: (385, 105)
    num_stacked_layers = 3

    lambda_l2_reg = 0.003
    GRADIENT_CLIPPING = 2.5
    # total_iteractions = total_iteractions
    KEEP_RATE = 0.5

    # order of features
    # input_features = []
    # for station in station_list :
    #     for feature in X_aq_list + X_meo_list:
    #         input_features.append(station + "_" + feature)
    # input_features.sort()
    output_features = []
    if city == 'ld':
        output_features = ['BL0_PM10', 'BL0_PM2.5', 'CD1_PM10', 'CD1_PM2.5', 'CD9_PM10', 'CD9_PM2.5', 'GN0_PM10',
                           'GN0_PM2.5', 'GN3_PM10', 'GN3_PM2.5', 'GR4_PM10', 'GR4_PM2.5', 'GR9_PM10', 'GR9_PM2.5',
                           'HV1_PM10', 'HV1_PM2.5', 'KF1_PM10', 'KF1_PM2.5', 'LW2_PM10', 'LW2_PM2.5', 'MY7_PM10',
                           'MY7_PM2.5', 'ST5_PM10', 'ST5_PM2.5', 'TH4_PM10', 'TH4_PM2.5']
    else:
        for station in station_list:
            for aq_feature in y_aq_list:
                output_features.append(station + "_" + aq_feature)

    output_features.sort()

    # 统计量值
    statistics = get_training_statistics(city)

    # Generate test KDD_CUP_2018 for the model

    # _, _, dev_y_original = SMAPE_on_dataset_v1(dev_y, dev_y, output_features, statistics, 1)  #
    # print(dev_x.shape, dev_y.shape)  # (36, 120, 104), (36, 48, 26))

    # Define training model
    rnn_model = build_graph(feed_previous=False,
                            input_seq_len=input_seq_len,
                            output_seq_len=output_seq_len,
                            hidden_dim=hidden_dim,
                            input_dim=input_dim,
                            output_dim=output_dim,
                            num_stacked_layers=num_stacked_layers,
                            learning_rate=learning_rate,
                            lambda_l2_reg=lambda_l2_reg,
                            GRADIENT_CLIPPING=GRADIENT_CLIPPING,
                            loss_function=loss_function)


    # model_name = '5 pre_days, 0 gap, huber_loss loss_function, multivariate_100_iteractions'
    X_predict = generate_X_test_set(city=city,
                                    station_list=station_list,
                                    X_aq_list=X_aq_list,
                                    X_meo_list=X_meo_list,
                                    pre_days=pre_days,
                                    gap=gap)
    # print(X_predict.shape)
    # 加载最好的模型
    # init = tf.global_variables_initializer()
    print(model_name)
    print(X_predict.shape)
    model_name = model_name
    # model_name = '5 pre_days, 0 gap, L2 loss_function, multivariate_75_iteractionsaver'
    with tf.Session() as sess:
        # sess.run(init)

        saver = rnn_model['saver']().restore(sess, os.path.join(model_path, model_name))
        # saver = rnn_model['saver']().restore(sess, os.path.join('result_2/ld', model_name))
        feed_dict = {rnn_model['enc_inp'][t]: X_predict[:, t, :] for t in range(input_seq_len)}  # batch prediction input_seq_len
        feed_dict.update(
            {rnn_model['target_seq'][t]: np.zeros([X_predict.shape[0], output_dim], dtype=np.float32) for t in
             range(output_seq_len)})
        final_test_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)

        final_test_preds = [np.expand_dims(pred, 1) for pred in final_test_preds]
        final_test_preds = np.concatenate(final_test_preds, axis=1)
        aver_smapes, _, model_preds_on_test = SMAPE_on_dataset_v1(final_test_preds, final_test_preds, output_features, statistics,
                                                        1)  # 仅仅是为了计算预测值

    # return aver_smapes_best, model_preds_on_dev, dev_y_original, model_preds_on_test, output_features  # 将在这种情况下表现最好的模型 的预测结果 和 模型的位置信息返回
    print("last result")
    print(aver_smapes)
    return aver_smapes, model_preds_on_test, output_features


if __name__   == '__main__':
    city = 'bj'
    day = 2
    aver_smapes_best, model_preds_on_test, output_features = train_and_dev(city,model_name="5 pre_days, 0 gap, L2 loss_function, multivariate_55_iteractions", pre_days=day)
    #aver_smapes_best, model_preds_on_dev, dev_y_original, model_preds_on_test, output_features = train_and_dev(city)
    print(output_features)
    print(model_preds_on_test.shape)
    # np.save("filename.npy", model_preds_on_test)
    # b = np.load("filename.npy")
    generator_result(model_preds_on_test, city, pre_days=day)

    # city = 'ld'
    # # 5 pre_days, 0 gap, L2 loss_function, multivariate_75_iteractionsaver
    # _, model_preds_on_test, output_features = train_and_dev(city, model_name="5 pre_days, 0 gap, huber_loss loss_function, multivariate_195_iteractions", pre_days=day)
    # # aver_smapes_best, model_preds_on_dev, dev_y_original, model_preds_on_test, output_features = train_and_dev(city)
    # print(output_features)
    # print(model_preds_on_test.shape)
    # # np.save("filename.npy", model_preds_on_test)
    # # b = np.load("filename.npy")
    # generator_result(model_preds_on_test, city, pre_days=5)
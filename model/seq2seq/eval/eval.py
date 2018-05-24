# -*- coding: utf-8 -*-
#  训练一个使用所有特征的模型
# import tkinter
import os
import sys
import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

# sys.path.append('../../metrics')

from utils.computer_offline_smapes import smape_score
from model.seq2seq.metrics import SMAPE_on_dataset_v1
from model.seq2seq.seq2seq_data_util import get_training_statistics, generate_eval_set
from model.seq2seq.seq2seq_model import build_graph
from model.seq2seq.generate_submission import generator_result

from utils.config import KddConfig

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

ld_station_list = ['BL0', 'CD1', 'CD9', 'GN0', 'GN3',
                   'GR4', 'GR9', 'HV1', 'KF1', 'LW2', 'MY7', 'ST5', 'TH4']
ld_X_aq_list = ['NO2', 'PM10', 'PM2.5']
ld_y_aq_list = ['PM10', 'PM2.5']
ld_X_meo_list = ["temperature", "pressure", "humidity",
                 "direction", "speed"]  # "wind_direction","wind_speed"


def eval(city='ld', pre_days=5, gap=0, loss_function="L2", model_name='', start_day=0):
    '''
    city='bj' or 'ld' : 针对某个城市的数据进行训练
    pre_days : 使用 pre_days 天数的数据进行预测
    gap : 0,12,24
        0 : 当天 23点以后进行的模型训练
        12 : 当天中午进行的模型训练
        24 : 不使用当天数据进行的训练
    loss_function : 使用不同的损失函数
    start_month: 要评估的数据开始的月份
    start_day: 要评估的数据开始的日期
    '''
    station_list = bj_station_list
    X_aq_list = bj_X_aq_list
    y_aq_list = bj_y_aq_list
    X_meo_list = bj_X_meo_list
    # model_path = '../result_2/0430/'
    if city == "bj":
        station_list = bj_station_list
        X_aq_list = bj_X_aq_list
        y_aq_list = bj_y_aq_list
        X_meo_list = bj_X_meo_list
        model_path = KddConfig.bj_model_path  # '../result_2/0430/'
    elif city == "ld":
        station_list = ld_station_list
        X_aq_list = ld_X_aq_list
        y_aq_list = ld_y_aq_list
        X_meo_list = ld_X_meo_list
        model_path = KddConfig.ld_model_path  # '../result_2/ld/'

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
    statistics = get_training_statistics(city, eval=True)

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

    X_eval, Y_eval = generate_eval_set(city=city,
                                       station_list=station_list,
                                       X_aq_list=X_aq_list,
                                       y_aq_list=ld_y_aq_list,
                                       X_meo_list=X_meo_list,
                                       pre_days=pre_days,
                                       gap=gap,
                                       start_day=start_day,
                                       )

    aver_smapes_best = 10
    model_preds_on_dev = None
    # model_name = ''
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        # print("Using checkpoint: ", name)
        saver = rnn_model['saver']().restore(
            sess, os.path.join(model_path, model_name))

        feed_dict = {rnn_model['enc_inp'][t]: X_eval[:, t, :]
                     for t in range(input_seq_len)}  # batch prediction
        feed_dict.update(
            {rnn_model['target_seq'][t]: np.zeros([X_eval.shape[0], output_dim], dtype=np.float32) for t in
             range(output_seq_len)})
        final_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)

        final_preds = [np.expand_dims(pred, 1) for pred in final_preds]
        final_preds = np.concatenate(final_preds, axis=1)
    print("Y_eval.shape:", Y_eval.shape,
          "final_preds.shape:", final_preds.shape)
    aver_smapes, smapes_of_features, forecast_original = SMAPE_on_dataset_v1(Y_eval, final_preds, output_features,
                                                                             statistics, 1)
    print(aver_smapes)

    return X_eval, Y_eval


if __name__ == '__main__':
    # city = 'bj'
    # aver_smapes_best, model_preds_on_test, output_features = train_and_dev(city,model_name="5 pre_days, 0 gap, L2 loss_function, multivariate_90_iteractions")
    # # aver_smapes_best, model_preds_on_dev, dev_y_original, model_preds_on_test, output_features = train_and_dev(city)
    # print(output_features)
    # print(model_preds_on_test.shape)
    # np.save("filename.npy", model_preds_on_test)
    # # b = np.load("filename.npy")
    # generator_result(model_preds_on_test, city)
    #
    # city = 'ld'
    # _, model_preds_on_test, output_features = train_and_dev(city, model_name="5 pre_days, 0 gap, huber_loss loss_function, multivariate_195_iteractions")
    # # aver_smapes_best, model_preds_on_dev, dev_y_original, model_preds_on_test, output_features = train_and_dev(city)
    # print(output_features)
    # print(model_preds_on_test.shape)
    # np.save("filename.npy", model_preds_on_test)
    # # b = np.load("filename.npy")
    # generator_result(model_preds_on_test, city)

    # X, Y = eval(city='ld', pre_days=5, gap=0, loss_function="L2",
    #             model_name='5 pre_days, 0 gap, L2 loss_function, multivariate_225_iteractions', start_day=0)
    X, Y = eval(city='ld', pre_days=5, gap=0, loss_function="L2",
                model_name='5 pre_days, 0 gap, L2 loss_function, multivariate_5_iteractions', start_day=0)

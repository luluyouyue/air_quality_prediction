# coding: utf-8
import pandas as pd
import numpy as np

def generator_result(model_preds, city='bj', pre_days=5):
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

    # ld_station_list = ['BL0','CD1','CD9','GN0','GN3','GR4','GR9','HV1','KF1','LW2','MY7','ST5','TH4']
    ld_station_list = ['CD1', 'BL0', 'GR4', 'MY7', 'HV1', 'GN3', 'GR9','LW2','GN0','KF1','CD9','ST5', 'TH4']
    ld_X_aq_list = ['NO2', 'PM10', 'PM2.5']
    ld_y_aq_list = ['PM2.5', 'PM10']
    ld_X_meo_list = ["temperature","pressure","humidity","direction","speed"]
    # submit = pd.read_csv('sample_submission.csv')
    submit = pd.read_csv('pre_days_5sample_submission.csv')

    if city == 'bj':
        # translate  downloaded  airQuality_id  to csv
        list = []
        name_0 = ['aotizhongxin_aq_O3', 'aotizhongxin_aq_PM10', 'aotizhongxin_aq_PM2.5', 'badaling_aq_O3', 'badaling_aq_PM10',
              'badaling_aq_PM2.5', 'beibuxinqu_aq_O3', 'beibuxinqu_aq_PM10', 'beibuxinqu_aq_PM2.5', 'daxing_aq_O3',
              'daxing_aq_PM10', 'daxing_aq_PM2.5', 'dingling_aq_O3', 'dingling_aq_PM10', 'dingling_aq_PM2.5', 'donggaocun_aq_O3',
              'donggaocun_aq_PM10', 'donggaocun_aq_PM2.5', 'dongsi_aq_O3', 'dongsi_aq_PM10', 'dongsi_aq_PM2.5', 'dongsihuan_aq_O3',
              'dongsihuan_aq_PM10', 'dongsihuan_aq_PM2.5', 'fangshan_aq_O3', 'fangshan_aq_PM10', 'fangshan_aq_PM2.5',
              'fengtaihuayuan_aq_O3', 'fengtaihuayuan_aq_PM10', 'fengtaihuayuan_aq_PM2.5', 'guanyuan_aq_O3', 'guanyuan_aq_PM10',
              'guanyuan_aq_PM2.5', 'gucheng_aq_O3', 'gucheng_aq_PM10', 'gucheng_aq_PM2.5', 'huairou_aq_O3', 'huairou_aq_PM10',
              'huairou_aq_PM2.5', 'liulihe_aq_O3', 'liulihe_aq_PM10', 'liulihe_aq_PM2.5', 'mentougou_aq_O3', 'mentougou_aq_PM10',
              'mentougou_aq_PM2.5', 'miyun_aq_O3', 'miyun_aq_PM10', 'miyun_aq_PM2.5', 'miyunshuiku_aq_O3', 'miyunshuiku_aq_PM10',
              'miyunshuiku_aq_PM2.5', 'nansanhuan_aq_O3', 'nansanhuan_aq_PM10', 'nansanhuan_aq_PM2.5', 'nongzhanguan_aq_O3',
              'nongzhanguan_aq_PM10', 'nongzhanguan_aq_PM2.5', 'pingchang_aq_O3', 'pingchang_aq_PM10', 'pingchang_aq_PM2.5',
              'pinggu_aq_O3', 'pinggu_aq_PM10', 'pinggu_aq_PM2.5', 'qianmen_aq_O3', 'qianmen_aq_PM10', 'qianmen_aq_PM2.5',
              'shunyi_aq_O3', 'shunyi_aq_PM10', 'shunyi_aq_PM2.5', 'tiantan_aq_O3', 'tiantan_aq_PM10', 'tiantan_aq_PM2.5',
              'tongzhou_aq_O3', 'tongzhou_aq_PM10', 'tongzhou_aq_PM2.5', 'wanliu_aq_O3', 'wanliu_aq_PM10', 'wanliu_aq_PM2.5',
              'wanshouxigong_aq_O3', 'wanshouxigong_aq_PM10', 'wanshouxigong_aq_PM2.5', 'xizhimenbei_aq_O3', 'xizhimenbei_aq_PM10',
              'xizhimenbei_aq_PM2.5', 'yanqin_aq_O3', 'yanqin_aq_PM10', 'yanqin_aq_PM2.5', 'yizhuang_aq_O3', 'yizhuang_aq_PM10',
              'yizhuang_aq_PM2.5', 'yongdingmennei_aq_O3', 'yongdingmennei_aq_PM10', 'yongdingmennei_aq_PM2.5', 'yongledian_aq_O3',
              'yongledian_aq_PM10', 'yongledian_aq_PM2.5', 'yufa_aq_O3', 'yufa_aq_PM10', 'yufa_aq_PM2.5', 'yungang_aq_O3',
              'yungang_aq_PM10', 'yungang_aq_PM2.5', 'zhiwuyuan_aq_O3', 'zhiwuyuan_aq_PM10', 'zhiwuyuan_aq_PM2.5']
        aq_shape = np.array(model_preds).shape
        for i in range(aq_shape[0]):
            for j in range(aq_shape[1]): # 48
                row = []
                for k in range(aq_shape[2]):  # 210

                    # 负值处理
                    if model_preds[i][j][k] < 0:
                        found_for = False
                        m = 0
                        while not found_for:
                            m += 1
                            if j - m <= 0:
                                for_row = 0
                                for_step = m
                                found_for = True
                                print('found_for')
                            else:
                               if j == 0 or model_preds[i][j-m][k] > 0:
                                    for_row = model_preds[i][j-m][k]
                                    for_step = m
                                    found_for = True
                                    print('found_for')

                        # 后边第几个是非空的
                        found_back = False
                        n = 0
                        while not found_back:
                            n += 1
                            if j + n >= 48:
                                back_row = 0
                                back_step = n
                                found_back = True
                                print('found_for')
                            else:
                                if j == 0 or model_preds[i][j+n][k] > 0:
                                    back_row = model_preds[i][j+n][k]
                                    back_step = n
                                    found_back = True
                                    print('found_for')
                        all_steps = for_step + back_step
                        delata_values = back_row - for_row
                        model_preds[i][j][k] = for_row + (for_step / all_steps) * delata_values
                    row.append(model_preds[i][j][k])
                list.append(row)
        name = []

        for station in station_list:
            for y in y_aq_list:
                name.append(station + '_' + y)

        airData = pd.DataFrame(columns=name_0, data=list)
        airData = airData[name]
        airData.head()
        predict = np.array(airData)
        predict_shape = predict.shape
        # predict[]

        new_predict = []
        for i in range(3):
            column = []
            for j in range(i, 105, 3):
               for k in range(48):
                   column.append(predict[k, j])

            new_predict.append(column)
        # print np.array(new_predict).shape

    # PM2.5,PM10,O3
        '''
        submit.loc[:1680, 'PM2.5'] = new_predict[0]
        submit.loc[:1680, 'PM10'] = new_predict[1]
        submit.loc[:1680, 'O3'] = new_predict[2]
        '''

        for i in range(1680):
            submit.iat[i, 1] = new_predict[0][i]

        for i in range(1680):
            submit.iat[i, 2] = new_predict[1][i]

        for i in range(1680):
            submit.iat[i, 3] = new_predict[2][i]

        # submit.to_csv('submission.csv')
    
    if city == 'ld':
        # translate  downloaded  airQuality_id  to csv
        list = []
        name_0 = ['BL0_PM10', 'BL0_PM2.5', 'CD1_PM10', 'CD1_PM2.5', 'CD9_PM10', 'CD9_PM2.5', 'GN0_PM10', 'GN0_PM2.5', 'GN3_PM10', 'GN3_PM2.5', 'GR4_PM10', 'GR4_PM2.5', 'GR9_PM10', 'GR9_PM2.5', 'HV1_PM10', 'HV1_PM2.5', 'KF1_PM10', 'KF1_PM2.5', 'LW2_PM10', 'LW2_PM2.5', 'MY7_PM10', 'MY7_PM2.5', 'ST5_PM10', 'ST5_PM2.5', 'TH4_PM10', 'TH4_PM2.5']
        aq_shape = np.array(model_preds).shape
        for i in range(aq_shape[0]):
            for j in range(aq_shape[1]):
                row = []
                for k in range(aq_shape[2]):
                    # print ('model_preds:', model_preds[i][j][k])
                    if model_preds[i][j][k] < 0:
                        found_for = False
                        m = 0
                        while not found_for:
                            m += 1
                            if j - m <= 0:
                                for_row = 0
                                for_step = m
                                found_for = True
                                print('found_for')
                            else:
                                if j == 0 or model_preds[i][j-m][k] > 0:
                                    for_row = model_preds[i][j-m][k]
                                    for_step = m
                                    found_for = True
                                    print('found_for')

                        # 后边第几个是非空的
                        found_back = False
                        n = 0
                        while not found_back:
                            n += 1
                            if j + n >= 48:
                                back_row = 0
                                back_step = n
                                found_back = True
                                print('found_for')
                            else:
                                if j == 0 or model_preds[i][j+n][k] > 0:
                                    back_row = model_preds[i][j+n][k]
                                    back_step = n
                                    found_back = True
                                    print('found_for')
                        all_steps = for_step + back_step
                        delata_values = back_row - for_row
                        model_preds[i][j][k] = for_row + (for_step / all_steps) * delata_values
                    row.append(model_preds[i][j][k])
                list.append(row)
        name = []

        for station in ld_station_list:
            for y in ld_y_aq_list:
                name.append(station + '_' + y)

        airData = pd.DataFrame(columns=name_0, data=list)
        airData = airData[name]
        airData.head()
        predict = np.array(airData)
        predict_shape = predict.shape
        # predict[]

        new_predict = []
        for i in range(2):
            column = []
            for j in range(i, 26, 2):
               for k in range(48):
                   column.append(predict[k, j])
            new_predict.append(column)
#        print np.array(new_predict).shape  # (1, 624)

    # PM2.5,PM10,O3
        '''
        submit.loc[:1680, 'PM2.5'] = new_predict[0]
        submit.loc[:1680, 'PM10'] = new_predict[1]
        submit.loc[:1680, 'O3'] = new_predict[2]
        '''

        for i in range(1680, 2304):
            # print new_predict[0][i-1680]
            submit.iat[i, 1] = new_predict[0][i-1680]

        for i in range(1680, 2304):
            submit.iat[i, 2] = new_predict[1][i-1680]

        for i in range(1680, 2304):
            submit.iat[i, 3] = 0

        print(submit.shape)
    if 'Unnamed: 0' in submit.columns:
        submit.drop('Unnamed: 0', axis=1, inplace=True)
    if 'Unnamed: 0' in submit.columns:
        submit.drop('Unnamed: 0', axis=1, inplace=True)

    if 'Unnamed: 0.1' in submit.columns:
        submit.drop('Unnamed: 0.1', axis=1, inplace=True)
    submit.to_csv('pre_days_'+str(pre_days)+'sample_submission.csv', index=False)
    #for row in range(35*48):

    #    submit[]
    # print(len(airData))
    # airData.to_csv('predict_bj_aq.csv')


    #print name









'''
f = open("bj_aq_20180331_20180427", 'r')
lines = f.readlines()
print(lines[1])
# print(len(f.readlines()))  # out: 0 ??  answer: the point of file is at the end of the file after the operation of f.readlines()

# name
line = lines[0]
items = line.strip().split(',')

# for j in xrange(len(items)):
    # name.append(items[j])
name = ['stationId', 'utc_time', 'PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
# list
for i in xrange(1, len(lines)):
    line = lines[i]
    items = line.strip().split(',')
    item = []
    for j in xrange(1, len(items)):
        item.append(items[j])
    list.append(item)

print('666')
airData = pd.DataFrame(columns=name, KDD_CUP_2018=list)
print(len(airData))
airData.to_csv('new_bj_aq.csv')
'''

##########################################################station_location.csv generator script#######################
'''
def translate_station_location_tocsv(raw_file, name):
    list = []
    with open(raw_file) as f:
    # fill list
        # assert 35 == len(f.readlines()), 'read file error!'
        print type(f)
        line = f.readline()
        while line:
            items = line.strip().split('\t')
            print len(items)
            assert 3 == len(items), 'split error!'
            item = []
            for j in xrange(len(items)):
                item.append(items[j])
            list.append(item)
            line = f.readline()
    print len(list)
    station_location = pd.DataFrame(columns=name, KDD_CUP_2018=list)
    station_location.to_csv('station_location.csv')


translate_station_location_tocsv('station.csv', name=['stationName', 'latitude', 'longitude'])
'''

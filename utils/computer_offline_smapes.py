import pandas as pd
import numpy as np

def smape(actual, predicted):
    a = np.abs(np.array(actual) - np.array(predicted))
    b = np.array(actual) + np.array(predicted)

    return 2 * np.mean(np.divide(a, b, out=np.zeros_like(a), where=b!=0, casting='unsafe'))

###### replace older bj station names #########

bj_station_dict = {'aotizhongx': 'aotizhongxin',
 'fengtaihua': 'fengtaihuayuan',
 'miyunshuik': 'miyunshuiku',
 'nongzhangu': 'nongzhanguan',
 'wanshouxig': 'wanshouxigong',
 'xizhimenbe': 'xizhimenbei',
 'yongdingme': 'yongdingmennei'}

def bj_station(s):
    if s[:s.find('_')] in bj_station_dict:
        return bj_station_dict[s[:s.find('_')]] + s[s.find('_'):]
    else:
        return s

########   read files  ########


def smape_score(sub_path, bj1_path, bj2_path, ld1_path, ld2_path):
# sub_path = 'auto_submission.csv'
# bj1_path = 'bj55.csv'
# bj2_path = 'bj56.csv'
# ld1_path = 'ld56.csv'      # Dont mind the filename. For some reasons, 'ld56' actually has 5-5 data of London > <
# ld2_path = 'ld57.csv'


####### read crawler file #######

    forecasted_station_list= ['CD1','BL0','GR4','MY7','HV1','GN3','GR9','LW2','GN0','KF1','CD9','ST5','TH4'] # filter valid stations
    tru_ld = pd.read_csv(ld2_path)


######## formatting ##########

    #tru_ld = tru_ld[tru_ld['date'] == day + 2]
    tru_ld = tru_ld[tru_ld['date'] == 20180506]
    tru_ld = tru_ld[tru_ld['stationId'].isin(forecasted_station_list)]
    tru_ld['hour'] = tru_ld['hour'] + 24

    tru_ld2 = pd.read_csv(ld1_path)
    #tru_ld2 = tru_ld2[tru_ld2['date'] == day + 1]
    tru_ld2 = tru_ld2[tru_ld2['date'] == 20180505]
    tru_ld2 = tru_ld2[tru_ld2['stationId'].isin(forecasted_station_list)]

    tru_ld = tru_ld2.append(tru_ld, ignore_index=True)
    tru_ld['stationId'] = tru_ld['stationId'] + '#' + tru_ld['hour'].apply(str)
    tru_ld.rename(columns = {'pm25': 'PM2.5', 'pm10': 'PM10', 'stationId': 'test_id'}, inplace = True)
    tru_ld.drop('date', axis = 1, inplace = True)
    tru_ld.drop('hour', axis = 1, inplace = True)
    tru_ld.drop('no2', axis = 1, inplace = True)

    ###### drop invalid data points ########
    tru_ld = tru_ld.dropna()
    tru_ld = tru_ld[(tru_ld['PM2.5'] >= 0)]
    tru_ld = tru_ld[(tru_ld['PM10'] >= 0)]

    ####### Similarly, generate ground truth dataframe for Beijing's stations  ########

    tru_bj = pd.read_csv(bj2_path)
    tru_bj['hour'] = tru_bj['hour'] + 24
    tru_bj2 = pd.read_csv(bj1_path)
    tru_bj = tru_bj2.append(tru_bj, ignore_index = True)

    tru_bj['test_id'] = tru_bj['stationId'] + '#' + tru_bj['hour'].apply(str)
    tru_bj.drop(['stationId', 'utc_time', 'hour', 'NO2', 'CO', 'SO2'], axis = 1, inplace = True)

    tru_bj = tru_bj.dropna()

    ##########  Read Submissions ###########

    sub = pd.read_csv(sub_path)
    sub = sub.fillna(0.0)

    sub_bj = sub.loc[sub['test_id'].str.find('#') != 3]

    sub_ld = sub.loc[sub['test_id'].str.find('#') == 3].drop('O3', axis=1)

    #########  Update older BJ station names #########
    sub_bj['asd'] = sub_bj['test_id'].apply(bj_station)
    sub_bj.drop('test_id', axis=1, inplace=True)
    sub_bj.rename(columns={'asd': 'test_id'}, inplace=True)

    ####### Merge into a resulting matrix #######
    result_ld = pd.merge(sub_ld, tru_ld, on = 'test_id', how = 'inner')
    result_bj = pd.merge(sub_bj, tru_bj, on = 'test_id', how = 'inner')

    ###### Calculate score ########
    b = smape([result_bj['PM2.5_x'], result_bj['PM10_x'], result_bj['O3_x']],
        [result_bj['PM2.5_y'], result_bj['PM10_y'], result_bj['O3_y']])
    l = smape([result_ld['PM2.5_x'], result_ld['PM10_x']], [result_ld['PM2.5_y'], result_ld['PM10_y']])
    score = (b + l) / 2

    print(score)
# -*- coding: utf-8 -*-
import requests
import datetime
from utils.config import KddConfig

# 获得当前的 UTC 时间
now = datetime.datetime.utcnow()

# 获得一天前和现在的时间
# year_s = ago.year
# month_s = ago.month
# day_s = ago.day
# hour_s = ago.hour

# year_e = now.year
# month_e = now.month
day = now.day
hour = now.hour

print("day is :", day, "hour is : ", hour)


start_month = 5
start_day = 1
end_month = 5
end_day = day + 1
print('generate KDD_CUP_2018 from', start_month,
      '-', start_day, 'to', end_month, '-', end_day)
# 下载数据
# '/home/competition/kdd2018/model/seq2seq/KDD_CUP_2018/test_data'
save_path = KddConfig.download_data_dir
url_bj_aq = 'https://biendata.com/competition/airquality/bj/2018-0%d-%d-0/2018-0%d-%d-23/2k0d1d8' % (
    start_month, start_day, end_month, end_day)
respones = requests.get(url_bj_aq)
name = save_path + "/Beijing/aq" + "/bj_aq_2018-0%d-%d-0_2018-0%d-%d-23.csv" % (
    start_month, start_day, end_month, end_day)
with open(name, 'w') as f:
    f.write(respones.text)
print("done ", name)


url_bj_meo = 'https://biendata.com/competition/meteorology/bj_grid/2018-0%d-%d-0/2018-0%d-%d-23/2k0d1d8' % (
    start_month, start_day, end_month, end_day)
respones = requests.get(url_bj_meo)
name = save_path+"/Beijing/grid_meo"+"/bj_grid_2018-0%d-%d-0_2018-0%d-%d-23.csv" % (
    start_month, start_day, end_month, end_day)
with open(name, 'w') as f:
    f.write(respones.text)
print("done ", name)


url_ld_aq = 'https://biendata.com/competition/airquality/ld/2018-0%d-%d-0/2018-0%d-%d-23/2k0d1d8' % (
    start_month, start_day, end_month, end_day)
respones = requests.get(url_ld_aq)
name = save_path + "/London/aq" + "/ld_aq_2018-0%d-%d-0_2018-0%d-%d-23.csv" % (
    start_month, start_day, end_month, end_day)
with open(name, 'w') as f:
    f.write(respones.text)
print("done ", name)


url_ld_meo = 'https://biendata.com/competition/meteorology/ld_grid/2018-0%d-%d-0/2018-0%d-%d-23/2k0d1d8' % (
    start_month, start_day, end_month, end_day)
respones = requests.get(url_ld_meo)
name = save_path+"/London/grid_meo"+"/ld_grid_2018-0%d-%d-0_2018-0%d-%d-23.csv" % (
    start_month, start_day, end_month, end_day)
with open(name, 'w') as f:
    f.write(respones.text)
print("done ", name)

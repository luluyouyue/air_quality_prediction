import pandas as pd


# translate  downloaded  airQuality_id  to csv
list = []
# name = []

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
airData = pd.DataFrame(columns=name, data=list)
print(len(airData))
airData.to_csv('new_bj_aq.csv')


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

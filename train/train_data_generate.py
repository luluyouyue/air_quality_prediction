from preprocess.aq_data_preprocess import aq_data_preprocess
from preprocess.weather_data_preprocess import meo_data_preprocess
from preprocess.train_dev_set_split import train_dev_set_split


if __name__ == '__main__':
    # print('generate bj ad data!')
    # aq_data_preprocess('bj')
    #
    # print('generate ld ad data!')
    # aq_data_preprocess('ld')
    #
    # print('generate bj meo data!')
    # meo_data_preprocess('bj')
    #
    # print('generate ld data!')
    # meo_data_preprocess('ld')

    print('split bj train data!')
    train_dev_set_split('bj')

    print('generate ld train data!')
    train_dev_set_split('ld')

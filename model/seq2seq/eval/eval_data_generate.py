from eval_aq_data_preprocess import aq_data_preprocess
from eval_weather_data_preprocess import meo_data_preprocess
# from model.seq2seq.weather_data_preprocess import  meo_data_preprocess
from model.seq2seq.train_dev_set_split import train_dev_set_split


if __name__ == '__main__':
    print('generate bj ad data!')
    aq_data_preprocess('bj')

    print('generate ld ad data!')
    aq_data_preprocess('ld')

    print('generate bj meo data!')
    meo_data_preprocess('bj')

    print('generate ld data!')
    meo_data_preprocess('ld')
    #
    print('split bj train data!')
    train_dev_set_split('bj', True)

    print('generate ld train data!')
    train_dev_set_split('ld', True)

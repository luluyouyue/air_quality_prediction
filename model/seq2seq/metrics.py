import numpy as np
from utils.config import KddConfig


def symmetric_mean_absolute_percentage_error(actual, forecast, y_mean, y_std):
    '''
    Compute the Symmetric mean absolute percentage error (SMAPE or sMAPE) on a single KDD_CUP_2018 of the dev set or test set.
    Details of SMAPE here : https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

    Args:
        actual : an actual values in the dev/test dataset.
        forecast : an model forecast values.
        y_mean : mean value used when doing preprocess.
        y_std : std value used when doing preprocess.

    '''
    actual = np.squeeze(actual)
    forecast = np.squeeze(forecast)

    assert len(actual) == len(
        forecast), "The shape of actual value and forecast value are not the same."

    length = len(actual)
    actual = actual * y_std + y_mean
    forecast = forecast * y_std + y_mean

    r = 0

    for i in range(length):
        f = forecast[i]
        a = actual[i]
        # print 'f:', f, 'a:', a
        r += abs(f-a) / ((abs(a)+abs(f))/2)

    return r/length, forecast


def SMAPE_on_dataset(actual_data, forecast_data, feature_list, y_mean, y_std, forecast_duration=24):
    '''
    Compute SMAPE value on the dataset of actual and forecast.

    Args:
        actual_data : actual KDD_CUP_2018 in the dev set or test set, shape [number_of_examples_in_the_dev_set, output_seq_length, num_of_output_features]
        forecast_data : forecast KDD_CUP_2018 which are predicted by the seq2seq model using x KDD_CUP_2018 in the dev set or test set, same shape as actual_data.
        forecast_duration : predict every 24 hours, so forecast_duration is set default to 1, because the dev set is sampled every 24 hours.
        feature_list : a list of features that is caculated in the forecast.
    Return:
        aver_smapes : average smape for all features on the test KDD_CUP_2018.
        smapes_of_features : smapes of different features on the test KDD_CUP_2018.
    '''
    assert actual_data.shape == forecast_data.shape, "The shape of actual KDD_CUP_2018 and perdiction KDD_CUP_2018 must match."

    number_of_features = actual_data.shape[2]
    smapes_list_of_features = {feature: [] for feature in feature_list}

    for i in range(0, actual_data.shape[0], forecast_duration):
        actual_data_item = actual_data[i]
        forecast_data_item = forecast_data[i]
        for j in range(number_of_features):
            feature = feature_list[j]
            a = actual_data_item[:, j]
            f = forecast_data_item[:, j]
            smape_a_feature_a_day = symmetric_mean_absolute_percentage_error(
                a, f, y_mean, y_std)
            smapes_list_of_features[feature].append(smape_a_feature_a_day)

    smapes_of_features = {feature: np.mean(
        value) for feature, value in smapes_list_of_features.items()}
    aver_smapes = np.mean(list(smapes_of_features.values()))

    return aver_smapes, smapes_of_features


# For new seq2seq model
def SMAPE_on_dataset_v1(actual_data, forecast_data, feature_list, statistics, forecast_duration=1):
    '''
    Compute SMAPE value on the dataset of actual and forecast.

    Args:
        actual_data : actual KDD_CUP_2018 in the dev set or test set, shape [number_of_examples_in_the_dev_set, output_seq_length, num_of_output_features]
        forecast_data : forecast KDD_CUP_2018 which are predicted by the seq2seq model using x KDD_CUP_2018 in the dev set or test set, same shape as actual_data.
        forecast_duration : predict every 24 hours, so forecast_duration is set default to 1, because the dev set is sampled every 24 hours.
        feature_list : a list of features that is caculated in the forecast. Need to be in the right order!!
        statistics : a pandas dataframe of statistics.
    Return:
        aver_smapes : average smape for all features on the test KDD_CUP_2018.
        smapes_of_features : smapes of different features on the test KDD_CUP_2018.
    '''
    assert actual_data.shape == forecast_data.shape, "The shape of actual KDD_CUP_2018 and perdiction KDD_CUP_2018 must match."
    log = open(KddConfig.actual_predict_log_path, 'w')
    log.write(','.join(feature_list)+'\n')
    for i in range(actual_data.shape[1]):
        log.write(str(actual_data[0][i])+'\n')
        log.write(str(forecast_data[0][i]) + '\n')

    forecast_original = np.zeros(forecast_data.shape)
    # print statistics.columns
    # print(feature_list)
    number_of_features = actual_data.shape[2]
    smapes_list_of_features = {feature: [] for feature in feature_list}

    # print(statistics.loc['mean']['CD1_PM10'])
    for i in range(0, actual_data.shape[0], forecast_duration):
        actual_data_item = actual_data[i]
        forecast_data_item = forecast_data[i]
        for j in range(number_of_features):
            feature = feature_list[j]
            # print(feature)
            a = actual_data_item[:, j]
            f = forecast_data_item[:, j]
            y_mean = statistics.loc['mean'][feature]
            y_std = statistics.loc['std'][feature]
            smape_a_feature_a_day, f_original_a_feature_a_day = symmetric_mean_absolute_percentage_error(
                a, f, y_mean, y_std)
            smapes_list_of_features[feature].append(smape_a_feature_a_day)
            forecast_original[i, :, j] = f_original_a_feature_a_day

    smapes_of_features = {feature: np.mean(
        value) for feature, value in smapes_list_of_features.items()}
    aver_smapes = np.mean(list(smapes_of_features.values()))

    return aver_smapes, smapes_of_features, forecast_original

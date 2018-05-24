class KddConfig(object):
    # project_dir = "/home/competition/kdd2018/"
    project_dir = "/home/competition/air_quality_prediction/"
    processed_data_dir = project_dir + "data/processed_data/"
    bj_aq_locations = project_dir + \
        "data/Beijing/location/Beijing_AirQuality_Stations_locations.xlsx"
    ld_aq_locations = project_dir + "data/London/location/London_AirQuality_Stations.csv"

    # evalmode = False
    # if evalmode:
    #     path_to_bj_aq = project_dir+"data/Newdata/Beijing/aq/"
    #     path_to_ld_aq = project_dir+"data/Newdata/London/aq/"
    #     path_to_bj_meo = project_dir+"data/Newdata/Beijing/grid_meo/"
    #     path_to_ld_meo = project_dir + "data/Newdata/London/grid_meo/"
    # else:
    path_to_bj_aq = project_dir+"data/Beijing/aq/"
    path_to_ld_aq = project_dir+"data/London/aq/"
    path_to_bj_meo = project_dir+"data/Beijing/grid_meo/"
    path_to_ld_meo = project_dir + "data/London/grid_meo/"

    eval_processed_data_dir = project_dir + "data/Newdata_processed/"
    path_to_new_bj_aq = project_dir+"data/Newdata/Beijing/aq/"
    path_to_new_ld_aq = project_dir+"data/Newdata/London/aq/"
    path_to_new_bj_meo = project_dir+"data/Newdata/Beijing/grid_meo/"
    path_to_new_ld_meo = project_dir + "data/Newdata/London/grid_meo/"

    path_to_model = project_dir+"model/seq2seq/result/"
    path_to_sample_submission = project_dir + "model/seq2seq/"

    dev_start_time = "2018-5-1 0:00"
    train_start_time = "2017/1/2 0:00"
    train_end_time = "2018/5/1 0:00"

    bj_model_path = project_dir + "model/seq2seq/trained_model/bj"
    ld_model_path = project_dir + "model/seq2seq/trained_model/ld"
    train_log_path = project_dir + "model/seq2seq/train_log.txt"
    actual_predict_log_path = project_dir + "model/seq2seq/actual_predict.txt"

    submission_pre5day_file_path = project_dir + \
        "submission/pre_days_5sample_submission.csv"
    submission_pre2day_file_path = project_dir + \
        "submission/pre_days_2sample_submission.csv"
    submission_file_path = project_dir + \
        "sample_submission.csv"
    submission_file_dir = project_dir + "submission/"

    download_data_dir = project_dir + "data/Newdata"

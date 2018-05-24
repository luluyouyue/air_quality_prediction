class KddConfig(object):
    # project_dir = "/home/competition/kdd2018/"
    project_dir = "/home/competition/air_quality_prediction/"
    test_data_dir = project_dir+"model/seq2seq/test_new/"
    # "./KDD_CUP_2018/test_data/Beijing/aq/"
    path_to_bj_aq = project_dir+"model/seq2seq/KDD_CUP_2018/test_data/Beijing/aq/"
    # "./KDD_CUP_2018/test_data/London/aq/"
    path_to_ld_aq = project_dir+"model/seq2seq/KDD_CUP_2018/test_data/London/aq/"
    # "./KDD_CUP_2018/Beijing/grid_meo/"
    path_to_bj_meo = project_dir+"model/seq2seq/KDD_CUP_2018/test_data/Beijing/grid_meo/"
    # ./KDD_CUP_2018/London/grid_meo/
    path_to_ld_meo = project_dir+"model/seq2seq/KDD_CUP_2018/test_data/London/grid_meo/"
    path_to_model = project_dir+"model/seq2seq/result/"
    path_to_sample_submission = project_dir+"model/seq2seq/"

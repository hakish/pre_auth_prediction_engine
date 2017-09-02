# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 08:43:35 2017

@author: kisho
"""
import os
from sklearn.model_selection import train_test_split
from scripts.model_builder import build_svm_model, build_nb_model
from scripts.preprocessor1 import do_preprocess_1
from scripts.util import read_csv_data, dummify, calc_perf_metrics_for_model
from scripts.predictons import predict
from scripts.preprocessor1 import do_preprocess_1


def test(test_data, target_col_name, model_file_path, plots_path, model_name):
    data_df_test = dummify(test_data, target_col_name)
    feature_frame_test = data_df_test.drop(target_col_name, axis=1)
    y_test = data_df_test[target_col_name]
    y_pred_test = predict(model_file_path, feature_frame_test)
    # Compute confusion matrix for test data predictions
    calc_perf_metrics_for_model(model_name, y_test, y_pred_test, target_col_name, plots_path, 'test')
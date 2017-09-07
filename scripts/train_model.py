# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 08:43:35 2017

@author: kisho
"""
from scripts.model_builder import *
from scripts.predictions import predict
from scripts.util import dummify, calc_perf_metrics_for_model, do_roc_analysis
from scipy.stats import describe
from numpy import unique
import logging

def train(train_data, target_col_name, plots_path, models_path):

    # As our dataset contains a lot of categorical variables we need to dummify them to run logistic Regression
    data_df_train = dummify(train_data, target_col_name)
    feature_frame_train = data_df_train.drop(target_col_name, axis=1)
    y_train = data_df_train[target_col_name]

    # Create various models to find patterns in data

    # ========================================================================================================= #
    # Build random forest model
    train_random_forest_model(feature_frame_train, target_col_name, data_df_train, y_train, plots_path, models_path)

    # Build neural network model
    train_neural_net_model(feature_frame_train, target_col_name, data_df_train, y_train, plots_path, models_path)

    # Build svm model
    train_svm_model(feature_frame_train, target_col_name, data_df_train, y_train, plots_path, models_path)

    # Build naives bayes model
    train_nb_model(feature_frame_train, target_col_name, data_df_train, y_train, plots_path, models_path)

    # Build Gradient Boosting model
    train_gbm_model(feature_frame_train, target_col_name, data_df_train, y_train, plots_path, models_path)
    # ========================================================================================================= #

def train_nb_model(feature_frame_train, target_col_name, train_data, y_train, plots_path, models_path):
    train_data_copy = train_data.copy()
    model_file_path = str(models_path + 'model_naive_bayes.sav')
    build_nb_model(train_data_copy, target_col_name, model_file_path)
    # Make predictions on the train data and check the model metrics
    y_pred_train, y_pred_prob_train = predict(model_file_path, feature_frame_train)
    # Compute confusion matrix on train data
    calc_perf_metrics_for_model('naive_bayes', y_train, y_pred_train, target_col_name,
                                plots_path, 'train')
    do_roc_analysis(y_train, y_pred_prob_train, 'naive_bayes', plots_path, 'train')

def train_svm_model(feature_frame_train, target_col_name, train_data, y_train, plots_path, models_path):
    train_data_copy = train_data.copy()
    model_file_path = str(models_path + 'model_svm.sav')
    build_svm_model(train_data_copy, target_col_name, model_file_path)
    # Make predictions on the train data and check the model metrics
    y_pred_train, y_pred_prob_train = predict(model_file_path, feature_frame_train)
    # Compute confusion matrix on train data
    calc_perf_metrics_for_model('svm', y_train, y_pred_train, target_col_name,
                                plots_path, 'train')
    do_roc_analysis(y_train, y_pred_prob_train, 'svm', plots_path, 'train')

def train_neural_net_model(feature_frame_train, target_col_name, train_data, y_train, plots_path, models_path):
        train_data_copy = train_data.copy()
        model_file_path = str(models_path + 'model_neural_net.sav')
        build_neural_network_model(train_data_copy, target_col_name, model_file_path)
        # Make predictions on the train data and check the model metrics
        y_pred_train, y_pred_prob_train = predict(model_file_path, feature_frame_train)
        # Compute confusion matrix on train data
        calc_perf_metrics_for_model('neural_net', y_train, y_pred_train, target_col_name,
                                    plots_path, 'train')
        do_roc_analysis(y_train, y_pred_prob_train, 'neural_net', plots_path, 'train')

def train_random_forest_model(feature_frame_train, target_col_name, train_data, y_train, plots_path, models_path):
    train_data_copy = train_data.copy()
    model_file_path = str(models_path + 'model_random_forest.sav')
    build_random_forest_model(train_data_copy, target_col_name, model_file_path)
    # Make predictions on the train data and check the model metrics
    y_pred_train, y_pred_prob_train = predict(model_file_path, feature_frame_train)
    logging.info("describe y_pred_train :: "+str(unique(y_pred_train)))
    # Compute confusion matrix on train data
    calc_perf_metrics_for_model('random_forest', y_train, y_pred_train, target_col_name,
                                plots_path, 'train')
    do_roc_analysis(y_train, y_pred_prob_train, 'random_forest', plots_path, 'train')

def train_gbm_model(feature_frame_train, target_col_name, train_data, y_train, plots_path, models_path):
    train_data_copy = train_data.copy()
    model_file_path = str(models_path + 'model_gbm.sav')
    build_gbm_model(train_data_copy, target_col_name, model_file_path)
    # Make predictions on the train data and check the model metrics
    y_pred_train, y_pred_prob_train = predict(model_file_path, feature_frame_train)
    logging.info("describe y_pred_train :: "+str(unique(y_pred_train)))
    # Compute confusion matrix on train data
    calc_perf_metrics_for_model('gbm', y_train, y_pred_train, target_col_name,
                                plots_path, 'train')
    do_roc_analysis(y_train, y_pred_prob_train, 'gbm', plots_path, 'train')

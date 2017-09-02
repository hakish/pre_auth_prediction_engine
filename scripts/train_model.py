# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 08:43:35 2017

@author: kisho
"""
from scripts.model_builder import *
from scripts.predictons import predict
from scripts.util import dummify, calc_perf_metrics_for_model


def train(train_data, target_col_name):

    # As our dataset contains a lot of categorical variables we need to dummify them to run logistic Regression
    data_df_train = dummify(train_data, target_col_name)
    feature_frame_train = data_df_train.drop(target_col_name, axis=1)
    y_train = data_df_train[target_col_name]

    # Create various models to find patterns in data

    # ========================================================================================================= #
    # Build random forest model
    train_random_forest_model(feature_frame_train, target_col_name, data_df_train, y_train)

    # Build neural network model
    train_neural_net_model(feature_frame_train, target_col_name, data_df_train, y_train)

    # Build svm model
    train_svm_model(feature_frame_train, target_col_name, data_df_train, y_train)

    # Build naives bayes model
    train_nb_model(feature_frame_train, target_col_name, data_df_train, y_train)
    # ========================================================================================================= #

def train_nb_model(feature_frame_train, target_col_name, train_data, y_train):
    train_data_copy = train_data.copy()
    build_nb_model(train_data_copy, target_col_name)
    # Make predictions on the train data and check the model metrics
    y_pred_train = predict('../models/preprocess1/model_naive_bayes.sav', feature_frame_train)
    # Compute confusion matrix on train data
    calc_perf_metrics_for_model('naive_bayes', y_train, y_pred_train, target_col_name,
                                '..\\plots\\preprocess1\\', 'train')

def train_svm_model(feature_frame_train, target_col_name, train_data, y_train):
    train_data_copy = train_data.copy()
    build_svm_model(train_data_copy, target_col_name)
    # Make predictions on the train data and check the model metrics
    y_pred_train = predict('../models/preprocess1/model_svm.sav', feature_frame_train)
    # Compute confusion matrix on train data
    calc_perf_metrics_for_model('svm', y_train, y_pred_train, target_col_name,
                                '..\\plots\\preprocess1\\', 'train')

def train_neural_net_model(feature_frame_train, target_col_name, train_data, y_train):
        train_data_copy = train_data.copy()
        build_neural_network_model(train_data_copy, target_col_name)
        # Make predictions on the train data and check the model metrics
        y_pred_train = predict('../models/preprocess1/model_neural_net.sav', feature_frame_train)
        # Compute confusion matrix on train data
        calc_perf_metrics_for_model('neural_net', y_train, y_pred_train, target_col_name,
                                    '..\\plots\\preprocess1\\', 'train')

def train_random_forest_model(feature_frame_train, target_col_name, train_data, y_train):
    train_data_copy = train_data.copy()
    build_random_forest_model(train_data_copy, target_col_name)
    # Make predictions on the train data and check the model metrics
    y_pred_train = predict('../models/preprocess1/model_random_forest.sav', feature_frame_train)
    # Compute confusion matrix on train data
    calc_perf_metrics_for_model('random_forest', y_train, y_pred_train, target_col_name,
                                '..\\plots\\preprocess1\\', 'train')
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 08:43:35 2017

@author: kisho
"""

from sklearn.model_selection import train_test_split

from scripts.ModelBuilder import build_random_forest_model, build_neural_network_model, build_svm_model
from scripts.preprocess1 import do_preprocess_1
from scripts.util import read_csv_data, dummify, calc_perf_metrics_for_model

filename = "K:\\insofe\\MyProject\\dataset\\PriorAuth_Data.csv"
columnNames = ["userid", "drug", "drugsubclass", "drugclass", "drugchemicalname", "gpi",
               "ndc", "druggroup", "doctorid", "rxgroupid", "bin", "pcn", "state", "transdate",
               "target"]

# Load in the data with `read_csv()`
pa_data_df = read_csv_data(filename, columnNames)

# Split the data into train and test
pa_data_df_train, pa_data_df_test = train_test_split(pa_data_df, test_size=0.33, random_state=42)

pa_data_df_train.shape
pa_data_df_test.shape
target_col_name = "target"
# See the summary statistics and plots
# descriptive_stats_and_plots(pa_data_df_train, plots_path='K:\\insofe\\MyProject\\plots\\')

# descriptive_stats_and_plots(pa_data_df_train, plots_path='K:\\insofe\\MyProject\\plots\\preprocess1\\')

# As our dataset contains a lot of categorical variables we need to dummify them to run logistic Regression
print("Do pre-process 1")
pa_data_df_train = do_preprocess_1(pa_data_df_train, target_col_name)
pa_data_df_train = dummify(pa_data_df_train, target_col_name)
feature_frame_train = pa_data_df_train.drop(target_col_name, axis=1)
y_train = pa_data_df_train[target_col_name]

pa_data_df_test = do_preprocess_1(pa_data_df_test, target_col_name)
pa_data_df_test = dummify(pa_data_df_test, target_col_name)
feature_frame_test = pa_data_df_test.drop(target_col_name, axis=1)
y_test = pa_data_df_test[target_col_name]
print("Do pre-process 1 .. ends", pa_data_df_train.head())

# do_logistic_regression(pa_data_df_train, target_col_name)

# Create various models to find patterns in data

# ========================================================================================================= #
# Do Random Forest Modelling
# pa_data_df_train_copy = pa_data_df_train.copy()
# rf_model = build_random_forest_model(pa_data_df_train_copy, target_col_name)
# print("Random Forest model is :: ", rf_model)
# # Make predictions on the train data and check the model metrics
#
# y_pred_train = rf_model.predict(feature_frame_train)
# # Compute confusion matrix on train data
# calc_perf_metrics_for_model(rf_model, 'Random_Forest', y_train, y_pred_train, target_col_name,
#                             'K:\\insofe\\MyProject\\plots\\preprocess1\\', 'train')
# # Make predictions on test data using random forest model
#
# y_pred_test = rf_model.predict(feature_frame_test)
#
# # Compute confusion matrix for test data predictions
# calc_perf_metrics_for_model(rf_model, 'Random_Forest', y_test, y_pred_test, target_col_name,
#                             'K:\\insofe\\MyProject\\plots\\preprocess1\\', 'test')
# plt.show()
# ========================================================================================================= #

# Build neural network model
# pa_data_df_train_copy = pa_data_df_train.copy()
# nn_model = build_neural_network_model(pa_data_df_train_copy, target_col_name)
# print("Neural Net model is :: ", nn_model)
# # Make predictions on the train data and check the model metrics
#
# y_pred_train = nn_model.predict(feature_frame_train)
# # Compute confusion matrix on train data
# calc_perf_metrics_for_model(nn_model, 'Neural_Net', y_train, y_pred_train, target_col_name,
#                             'K:\\insofe\\MyProject\\plots\\preprocess1\\', 'train')
# # Make predictions on test data using random forest model
#
# y_pred_test = nn_model.predict(feature_frame_test)
#
# # Compute confusion matrix for test data predictions
# calc_perf_metrics_for_model(nn_model, 'Neural_Net', y_test, y_pred_test, target_col_name,
#                             'K:\\insofe\\MyProject\\plots\\preprocess1\\', 'test')
# ========================================================================================================= #

# Build svm model
pa_data_df_train_copy = pa_data_df_train.copy()
svm_model = build_svm_model(pa_data_df_train_copy, target_col_name)
print("svm model is :: ", svm_model)
# Make predictions on the train data and check the model metrics

y_pred_train = svm_model.predict(feature_frame_train)
# Compute confusion matrix on train data
calc_perf_metrics_for_model(svm_model, 'SVM', y_train, y_pred_train, target_col_name,
                            'K:\\insofe\\MyProject\\plots\\preprocess1\\', 'train')
# Make predictions on test data using random forest model

y_pred_test = svm_model.predict(feature_frame_test)

# Compute confusion matrix for test data predictions
calc_perf_metrics_for_model(svm_model, 'SVM', y_test, y_pred_test, target_col_name,
                            'K:\\insofe\\MyProject\\plots\\preprocess1\\', 'test')
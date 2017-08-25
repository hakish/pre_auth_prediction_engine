# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 08:43:35 2017

@author: kisho
"""
# Import the `pandas` library as `pd`
import pandas as pd
from sklearn.model_selection import train_test_split
from scripts.descriptive_stats_and_plots import descriptive_stats_and_plots
from scripts.util import read_csv_data, dummify, create_plot_confusion_matrix, check_col_distribution_and_plot
from scripts.preprocess1 import do_preprocess_1
from scripts.LogisticRegression import do_logistic_regression
from scripts.RandomForestForPAPE import create_random_forest_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score

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
print("Do pre-process 1 .. ends", pa_data_df_train.head())
# do_logistic_regression(pa_data_df_train, target_col_name)
# Create various models to find patterns in data
# Do Random Forest Modelling
rf_model = create_random_forest_model(pa_data_df_train, target_col_name)
print("Random Forest model is :: ", rf_model)
# Make predictions on the train data and check the model metrics
feature_frame_train = pa_data_df_train.drop(target_col_name, axis=1)
y_train = pa_data_df_train[target_col_name]
y_pred_train = rf_model.predict(feature_frame_train)
# Compute confusion matrix
print("Compute the confusion for predictions on train data")
create_plot_confusion_matrix(y_train, y_pred_train, 'K:\\insofe\\MyProject\\plots\\')

# Make predictions on test data using random forest model
pa_data_df_test = do_preprocess_1(pa_data_df_test, target_col_name)
pa_data_df_test = dummify(pa_data_df_test, target_col_name)
feature_frame_test = pa_data_df_test.drop(target_col_name, axis=1)
y_test = pa_data_df_test[target_col_name]
y_pred_test = rf_model.predict(feature_frame_test)
# Compute confusion matrix
print("Look at the distribution of the response in test data")
print(y_test.describe())


# Look at the count of the target variable in the dataset
check_col_distribution_and_plot(y_test, 'K:\\insofe\\MyProject\\plots\\')


create_plot_confusion_matrix(y_test, y_pred_test, 'K:\\insofe\\MyProject\\plots\\')

print("Accuracy of RF model is ::")
print(accuracy_score(y_test, y_pred_test))
print("Recall of RF model is ::")
print(recall_score(y_test, y_pred_test))
# plt.show()
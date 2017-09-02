# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 08:43:35 2017

@author: kisho
"""
from sklearn.model_selection import train_test_split

from scripts.test_model import test
from scripts.train_model import train
from scripts.util import read_csv_data
from scripts.preprocessor1 import do_preprocess_1

filename = "K:\\insofe\\MyProject\\dataset\\PriorAuth_Data.csv"
columnNames = ["userid", "drug", "drugsubclass", "drugclass", "drugchemicalname", "gpi",
               "ndc", "druggroup", "doctorid", "rxgroupid", "bin", "pcn", "state", "transdate",
               "target"]

# Load in the data with `read_csv()`
pa_data_df = read_csv_data(filename, columnNames)

# Split the data into train and test
pa_data_df_train, pa_data_df_test = train_test_split(pa_data_df, test_size=0.33, random_state=42)
print(pa_data_df_train.shape)
print(pa_data_df_test.shape)
target_col_name = "target"
# See the summary statistics and plots
# descriptive_stats_and_plots(pa_data_df_train, plots_path='K:\\insofe\\MyProject\\plots\\')

# descriptive_stats_and_plots(pa_data_df_train, plots_path='K:\\insofe\\MyProject\\plots\\preprocess1\\')

# =================================================================================================================#
# 1.
# Pre processing approach:
# Reducing the dimensionality of the dataset due to large number of levels in categorical data
# by taking the top 10 levels by frequency of occurence and marking the rest as others.
# =================================================================================================================#
print("Preprocess the train data and create various models")
# Do some pre-processing on the data
train_data_pre_processed = do_preprocess_1(pa_data_df_train, target_col_name)
train(train_data_pre_processed, target_col_name)
print("training complete...")

print("evaluate the models on the test data")
test_data_pre_processed = do_preprocess_1(pa_data_df_test, target_col_name)
test(test_data_pre_processed.copy(), target_col_name, '../models/preprocess1/model_naive_bayes.sav', '../plots/preprocess1/',
     'naive_bayes')
test(test_data_pre_processed.copy(), target_col_name, '../models/preprocess1/model_random_forest.sav', '../plots/preprocess1/',
     'random_forest')
test(test_data_pre_processed.copy(), target_col_name, '../models/preprocess1/model_neural_net.sav', '../plots/preprocess1/',
     'neural_net')
test(test_data_pre_processed.copy(), target_col_name, '../models/preprocess1/model_svm.sav', '../plots/preprocess1/', 'svm')
print("testing complete...")
# =================================================================================================================#

# =================================================================================================================#
# 2.
# Pre processing approach:
# Reducing the dimensionality of the dataset due to large number of levels in categorical data
# by applying clustering to find similarity in the levels.
# =================================================================================================================#
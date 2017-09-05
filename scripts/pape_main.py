# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 08:43:35 2017

@author: kisho
"""
# =================================================================================================================#
# Import modules and functions
# =================================================================================================================#
from sklearn.model_selection import train_test_split
import logging
from scripts.test_model import test
from scripts.train_model import train
from scripts.util import read_csv_data
from scripts.preprocessor import do_preprocess_1, do_preprocess_2
# =================================================================================================================#

# Initialize the loggers
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)
fileHandler = logging.FileHandler("{0}/{1}.log".format('../logs/', 'pape_log'))
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

filename = "../dataset/PriorAuth_Data.csv"
columnNames = ["userid", "drug", "drugsubclass", "drugclass", "drugchemicalname", "gpi",
               "ndc", "druggroup", "doctorid", "rxgroupid", "bin", "pcn", "state", "transdate",
               "target"]

# Load in the data with `read_csv()`
pa_data_df = read_csv_data(filename, columnNames)

# Split the data into train and test
pa_data_df_train, pa_data_df_test = train_test_split(pa_data_df, test_size=0.33, random_state=42)
logging.info(pa_data_df_train.shape)
logging.info(pa_data_df_test.shape)
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
logging.info("Preprocess the train data with approach 1 and create various models")
# Do some pre-processing on the data
train_data_pre_processed = do_preprocess_1(pa_data_df_train.copy(), target_col_name)
train(train_data_pre_processed, target_col_name, '../plots/preprocess1/', '../models/preprocess1/')
logging.info("training complete...")

logging.info("evaluate the models on the test data")
test_data_pre_processed = do_preprocess_1(pa_data_df_test.copy(), target_col_name)
test(test_data_pre_processed.copy(), target_col_name, '../models/preprocess1/model_naive_bayes.sav',
     '../plots/preprocess1/',
     'naive_bayes')
test(test_data_pre_processed.copy(), target_col_name, '../models/preprocess1/model_random_forest.sav',
     '../plots/preprocess1/',
     'random_forest')
test(test_data_pre_processed.copy(), target_col_name, '../models/preprocess1/model_neural_net.sav',
     '../plots/preprocess1/',
     'neural_net')
test(test_data_pre_processed.copy(), target_col_name, '../models/preprocess1/model_svm.sav', '../plots/preprocess1/',
     'svm')
logging.info("testing complete...")
# =================================================================================================================#

# =================================================================================================================#
# 2.
# Pre processing approach:
# Reducing the dimensionality of the dataset due to large number of levels in categorical data
# by applying clustering to find similarity in the levels.
# =================================================================================================================#
logging.info("Preprocess the train data with approach 2 and create various models")
# Do some pre-processing on the data
train_data_pre_processed = do_preprocess_2(pa_data_df_train.copy(), target_col_name)
logging.info(train_data_pre_processed.describe())
train(train_data_pre_processed, target_col_name, '../plots/preprocess2/', '../models/preprocess2/')
logging.info("training complete...")

logging.info("evaluate the models on the test data")
test_data_pre_processed = do_preprocess_2(pa_data_df_test.copy(), target_col_name)
logging.info("test data pre-processed describe :: "+str(test_data_pre_processed.describe()))
test(test_data_pre_processed.copy(), target_col_name, '../models/preprocess2/model_naive_bayes.sav',
     '../plots/preprocess2/',
     'naive_bayes')
test(test_data_pre_processed.copy(), target_col_name, '../models/preprocess2/model_random_forest.sav',
     '../plots/preprocess2/',
     'random_forest')
test(test_data_pre_processed.copy(), target_col_name, '../models/preprocess2/model_neural_net.sav',
     '../plots/preprocess2/',
     'neural_net')
test(test_data_pre_processed.copy(), target_col_name, '../models/preprocess2/model_svm.sav', '../plots/preprocess2/',
     'svm')
logging.info("testing complete...")
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  21 08:43:35 2017

@author: kisho
"""

# Import the `pandas` library as `pd`
import pandas as pd
import numpy as np
from scripts.util import drop_cols
import matplotlib.pyplot as plt

# Function definitions
def doPredictorTagetFrequencyTop10Levels(predictorColName, df):
    df2 = df.copy()
    df2['target2'] =  np.where(df2['target']!='', '1', '0')
    predictorTargetFreqTotTab = pd.crosstab(index=df2[predictorColName], columns=df2["target2"])
    predictorTargetFreqTotTab.describe()
    top10levels = predictorTargetFreqTotTab.nlargest(10, '1').copy()
    # print(top10levels.index.get_values())
    top10levels = top10levels.index.get_values()
    print("top10levels for "+predictorColName)
    print(top10levels)
    # df[predictorColName] = np.where(df[predictorColName] not in top10levels, 'other', df[predictorColName])
    # print(df[predictorColName])
    reduceDim = lambda x: (x if x in top10levels else 'other')
    print(reduceDim)
    df[predictorColName] = df[predictorColName].apply(reduceDim)
    return df

def doPreprocessWithTop10Levels(df):
    df = doPredictorTagetFrequencyTop10Levels("drug", df)
    df = doPredictorTagetFrequencyTop10Levels("drugclass", df)
    df = doPredictorTagetFrequencyTop10Levels("drugsubclass", df)
    df = doPredictorTagetFrequencyTop10Levels("drugchemicalname", df)
    df = doPredictorTagetFrequencyTop10Levels("druggroup", df)
    df = doPredictorTagetFrequencyTop10Levels("state", df)
    df = doPredictorTagetFrequencyTop10Levels("pcn", df)
    df = doPredictorTagetFrequencyTop10Levels("rxgroupid", df)
    df = doPredictorTagetFrequencyTop10Levels("bin", df)
    df = doPredictorTagetFrequencyTop10Levels("ndc", df)
    df = doPredictorTagetFrequencyTop10Levels("gpi", df)
    return df

def do_preprocess_1(df, target_col_name):
    # Do data pre-processing with first approach of dimensionality reduction by taking
    # the top 10 levels by frequency for each attribute and marking everything else as others.
    # Drop a few columns as they should not have any influence on the target
    pa_data_df_train = drop_cols(df, ["userid", "doctorid", "transdate"])
    print(pa_data_df_train.describe())
    print(pa_data_df_train.dtypes)
    # Since all the columns are categorical attributes convert to the appropriate
    # categorical type
    for col in pa_data_df_train.columns:
        pa_data_df_train[col] = pa_data_df_train[col].astype('category')

    pa_data_df_train = doPreprocessWithTop10Levels(pa_data_df_train)
    toZeroOrOne = lambda x: (1 if x == 'TRUE' else 0)
    pa_data_df_train[target_col_name] = pa_data_df_train[target_col_name].apply(toZeroOrOne)
    print("Pre-processed data frame :: ")
    print(pa_data_df_train.describe())
    print(pa_data_df_train.dtypes)
    return df
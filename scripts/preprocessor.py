# -*- coding: utf-8 -*-
"""
Created on Mon Aug  21 08:43:35 2017

@author: kisho
"""

# Import the `pandas` library as `pd`
import pandas as pd
import numpy as np
from scripts.util import drop_cols
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from scripts.descriptive_stats_and_plots import doPredictorTagetFrequencyTotalPlot
from kmodes import kmodes
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

def do_preprocess_2(df, target_col_name):
    # ------------------------------------------------------------------------------------ #
    # Do data pre-processing with second approach of dimensionality reduction by taking
    # finding the similarity between various levels with clustering technique and using
    # the cluster numbers instead of the actual levels.
    # ------------------------------------------------------------------------------------ #
    # Drop a few columns as they should not have any influence on the target
    # ------------------------------------------------------------------------------------ #

    df = drop_cols(df, ["userid", "doctorid", "transdate"])
    print(df.describe())
    print(df.dtypes)
    # Since all the columns are categorical attributes convert to the appropriate
    # categorical type
    for col in df.columns:
        df[col] = df[col].astype('category')

    print("Pre-processed data frame :: ")
    print(df.describe())
    print(df.dtypes)
    feature_frame_train = df.drop(target_col_name, axis=1)
    # check_optimum_cluster_num(feature_frame_train)
    km = kmodes.KModes(n_clusters=5, init='Huang', n_init=5, verbose=1)
    clusters = km.fit_predict(feature_frame_train)
    df['cluster_num'] = clusters
    df['cluster_num'] = df['cluster_num'].astype('category')
    # silhouette_score = metrics.silhouette_score(feature_frame_train, labels, metric='cosine')
    print("Evaluate the clustering algorithm..")
    # print("silhouette score = "+str(silhouette_score))
    # The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering.
    # Scores around zero indicate overlapping clusters.
    # The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.
    print(df.describe())
    print(df.dtypes)
    doPredictorTagetFrequencyTotalPlot('cluster_num', df, '../plots/preprocess2/')
    trunc_df = df[['cluster_num', target_col_name]]
    print(trunc_df.describe())
    print(trunc_df.dtypes)
    toZeroOrOne = lambda x: (1 if x == 'TRUE' else 0)
    trunc_df[target_col_name] = trunc_df[target_col_name].apply(toZeroOrOne)
    trunc_df[target_col_name] = trunc_df[target_col_name].astype('category')
    return trunc_df


def check_optimum_cluster_num(feature_frame_train):
    cluster_num_cost = {}
    for num_of_clusters in range(1, 21):
        print('Number of clusters :: ', num_of_clusters)
        km = kmodes.KModes(n_clusters=num_of_clusters, init='Huang', n_init=20, verbose=1)
        clusters = km.fit_predict(feature_frame_train)
        cluster_num_cost.update({num_of_clusters: km.cost_})
        print('Cluster cost is :: ', km.cost_)
    print(cluster_num_cost)
    plt.bar(range(len(cluster_num_cost)), cluster_num_cost.values(), align='center')
    plt.xticks(range(len(cluster_num_cost)), cluster_num_cost.keys())
    plt.savefig('../plots/preprocess2/' + 'cluster_num_cost_itr' + "_" + '20' + ".png")
    plt.show()
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  21 08:43:35 2017

@author: kisho
"""

# Import the `pandas` library as `pd`
import pandas as pd
import numpy as np
from scripts.util import drop_cols
import prince
import logging
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from scripts.descriptive_stats_and_plots import doPredictorTagetFrequencyTotalPlot
from kmodes import kmodes
import matplotlib.pyplot as plt


# Function definitions
def doPredictorTagetFrequencyTop10Levels(predictorColName, df):
    df2 = df.copy()
    df2['target2'] = np.where(df2['target'] != '', '1', '0')
    predictorTargetFreqTotTab = pd.crosstab(index=df2[predictorColName], columns=df2["target2"])
    predictorTargetFreqTotTab.describe()
    top10levels = predictorTargetFreqTotTab.nlargest(10, '1').copy()
    # logging.info(top10levels.index.get_values())
    top10levels = top10levels.index.get_values()
    logging.info("top10levels for " + predictorColName)
    logging.info(top10levels)
    # df[predictorColName] = np.where(df[predictorColName] not in top10levels, 'other', df[predictorColName])
    # logging.info(df[predictorColName])
    reduceDim = lambda x: (x if x in top10levels else 'other')
    logging.info(reduceDim)
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
    logging.info(pa_data_df_train.describe())
    logging.info(pa_data_df_train.dtypes)
    # Since all the columns are categorical attributes convert to the appropriate
    # categorical type
    for col in pa_data_df_train.columns:
        pa_data_df_train[col] = pa_data_df_train[col].astype('category')

    pa_data_df_train = doPreprocessWithTop10Levels(pa_data_df_train)
    toZeroOrOne = lambda x: (1 if x == 'TRUE' else 0)
    pa_data_df_train[target_col_name] = pa_data_df_train[target_col_name].apply(toZeroOrOne)
    logging.info("Pre-processed data frame :: ")
    logging.info(pa_data_df_train.describe())
    logging.info(pa_data_df_train.dtypes)
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
    logging.info(df.describe())
    logging.info(df.dtypes)
    # Since all the columns are categorical attributes convert to the appropriate
    # categorical type
    for col in df.columns:
        df[col] = df[col].astype('category')

    logging.info("Pre-processed data frame :: ")
    logging.info(df.describe())
    logging.info(df.dtypes)
    feature_frame_train = df.drop(target_col_name, axis=1)
    # Get optimum levels for each categorical attribute
    updated_df = get_cluster_num_for_levels(df, target_col_name)
    # # check_optimum_cluster_num(feature_frame_train)
    # km = kmodes.KModes(n_clusters=5, init='Huang', n_init=5, verbose=1)
    # clusters = km.fit_predict(feature_frame_train)
    # df['cluster_num'] = clusters
    # df['cluster_num'] = df['cluster_num'].astype('category')
    # # silhouette_score = metrics.silhouette_score(feature_frame_train, labels, metric='cosine')
    # logging.info("Evaluate the clustering algorithm..")
    # # logging.info("silhouette score = "+str(silhouette_score))
    # # The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering.
    # # Scores around zero indicate overlapping clusters.
    # # The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.
    # logging.info(df.describe())
    # logging.info(df.dtypes)
    # doPredictorTagetFrequencyTotalPlot('cluster_num', df, '../plots/preprocess2/')
    # trunc_df = df[['cluster_num', target_col_name]]
    # logging.info(trunc_df.describe())
    # logging.info(trunc_df.dtypes)
    toZeroOrOne = lambda x: (1 if x == 'TRUE' else 0)
    # trunc_df[target_col_name] = trunc_df[target_col_name].apply(toZeroOrOne)
    # trunc_df[target_col_name] = trunc_df[target_col_name].astype('category')
    logging.info('updated_df :: ' + str(updated_df.describe()))
    updated_df[target_col_name] = updated_df[target_col_name].apply(toZeroOrOne)
    updated_df[target_col_name] = updated_df[target_col_name].astype('category')
    updated_df = updated_df.drop(feature_frame_train.columns, axis=1)
    return updated_df


def get_cluster_num_for_levels(data_df, target_col_name):
    for column in data_df.columns:
        if column != target_col_name:
            cluster_num_cost = {}
            feature_frame_train = data_df[[column, target_col_name]]
            logging.info(str('feature_frame_train is  :: ' + str(feature_frame_train.describe())))
            # for num_of_clusters in range(1, 5):
            #     logging.info('Number of clusters :: ' + str(num_of_clusters))
            #     km = kmodes.KModes(n_clusters=num_of_clusters, init='Huang', n_init=1, verbose=1)
            #     clusters = km.fit_predict(feature_frame_train)
            #     cluster_num_cost.update({num_of_clusters: km.cost_})
            #     logging.info('Cluster cost is :: ' + str(km.cost_))
            km = kmodes.KModes(n_clusters=4, init='Huang', n_init=5, verbose=1)
            clusters = km.fit_predict(feature_frame_train)
            cluster_num_cost.update({4: km.cost_})
            cluster_col = column + '_cluster_num'
            data_df[cluster_col] = clusters
            data_df[cluster_col] = data_df[cluster_col].astype('category')
            logging.info(cluster_num_cost)
            plt.bar(range(len(cluster_num_cost)), cluster_num_cost.values(), align='center')
            plt.xticks(range(len(cluster_num_cost)), cluster_num_cost.keys())
            plt.ylabel('Cost')
            plt.xlabel('Number of clusters')
            plt.savefig('../plots/preprocess2/' + cluster_col + "_" + '20' + ".png")
            plt.gcf().clear()
    # plt.show()
    return data_df

    # def check_optimum_cluster_num(data_df):
    #     cluster_num_cost = {}
    #     for num_of_clusters in range(1, 21):
    #         logging.info('Number of clusters :: ', num_of_clusters)
    #         km = kmodes.KModes(n_clusters=num_of_clusters, init='Huang', n_init=20, verbose=1)
    #         clusters = km.fit_predict(data_df)
    #         cluster_num_cost.update({num_of_clusters: km.cost_})
    #         logging.info('Cluster cost is :: ', km.cost_)
    #     logging.info(cluster_num_cost)
    #     plt.bar(range(len(cluster_num_cost)), cluster_num_cost.values(), align='center')
    #     plt.xticks(range(len(cluster_num_cost)), cluster_num_cost.keys())
    #     plt.savefig('../plots/preprocess2/' + 'cluster_num_cost_itr' + "_" + '20' + ".png")
    #     plt.show()

def do_preprocess_3(df, target_col_name, plots_path, suffix):
    # ------------------------------------------------------------------------------------ #
    # Do data pre-processing with second approach of dimensionality reduction by taking
    # finding the similarity between various levels with clustering technique and using
    # the cluster numbers instead of the actual levels.
    # ------------------------------------------------------------------------------------ #
    # Drop a few columns as they should not have any influence on the target
    # ------------------------------------------------------------------------------------ #

    df = drop_cols(df, ["userid", "doctorid", "transdate"])
    logging.info(df.describe())
    logging.info(df.dtypes)
    # Since all the columns are categorical attributes convert to the appropriate
    # categorical type
    for col in df.columns:
        df[col] = df[col].astype('category')

    logging.info('Dimensionality Reduction with MCA')
    mca = prince.MCA(df, n_components=1100)
    # logging.info('principal components are :: '+str(mca.categorical_columns))
    # logging.info('principal components are :: ' + str(mca.column_component_contributions))
    # logging.info('principal components are :: ' + str(mca.column_correlations))
    # logging.info('principal components are :: ' + str(mca.column_cosine_similarities))
    print('MCA is :: ', mca)
    logging.info('principal components are :: ' + str(mca.eigenvalues))
    logging.info('column_correlations are :: ' + str(mca.column_correlations))
    logging.info('cumulative_explained_inertia are :: ' + str(mca.cumulative_explained_inertia))
    logging.info('explained_inertia are :: ' + str(mca.explained_inertia))
    logging.info('cumulative_explained_inertia are :: ' + str(mca.row_cosine_similarities))
    logging.info(' row_principal_coordinates are :: ' + str(mca. row_principal_coordinates))
    # logging.info('principal components are :: ' + str(mca.column_standard_coordinates))
    # mca.plot_rows(show_points=True, show_labels=False, ellipse_fill=True)
    # mca.plot_relationship_square()
    mca.plot_cumulative_inertia(threshold=0.8)
    # plt.savefig(str(plots_path) + 'MCA_Analysis_Cumulative_Inertia_'+suffix + '.png')
    print(mca.head())
    logging.info("Pre-processed data frame :: ")
    logging.info(df.describe())
    logging.info(df.dtypes)
    #feature_frame_train = df.drop(target_col_name, axis=1)
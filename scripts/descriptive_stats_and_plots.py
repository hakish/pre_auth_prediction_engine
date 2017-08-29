# -*- coding: utf-8 -*-
"""
Created on Mon Aug  21 08:43:35 2017

@author: kisho
"""

# Import the `pandas` library as `pd`
import pandas as pd
import matplotlib.pyplot as plt

# Function definitions

def doPredictorTagetFrequencyPlot(predictorColName, df, plots_path):
    predictorTargetFreqTab = pd.crosstab(index=df[predictorColName], columns=df["target"]).copy()
    top10PredictorClassPASuccess = predictorTargetFreqTab.nlargest(10, 'TRUE').copy()
    del top10PredictorClassPASuccess['FALSE']
    top10PredictorClassPASuccess.plot(kind="bar", figsize=(8, 8), stacked=True, legend=True,
                              title="Top 10 "+predictorColName+" with PA Success")
    plt.savefig(plots_path + predictorColName + "_" + "TargetFreqPlotSuccess.png")
    top10PredictorClassPAFail = predictorTargetFreqTab.nlargest(10, 'FALSE').copy()
    del top10PredictorClassPAFail['TRUE']
    top10PredictorClassPAFail.plot(kind="bar", figsize=(8, 8), stacked=True, legend=True,color='red',
                              title="Top 10 "+predictorColName+" with PA Failure")
    plt.savefig(plots_path + predictorColName + "_" + "TargetFreqPlotFail.png")

def doPredictorTagetFrequencyTotalPlot(predictorColName, df, plots_path):
    predictorTargetFreqTotTab = pd.crosstab(index=df[predictorColName], columns=df["target"]).copy()
    predictorTargetFreqTotTab.plot(kind="bar", figsize=(8, 8), stacked=True, legend=True,
                                   title="Frequency Plot for "+predictorColName)
    plt.savefig(plots_path + predictorColName + "_" + "TargetFreqTotalPlot.png")

def doPredictorTagetFrequencyProbPlot(predictorColName, df, plots_path):
    predictorTargetFreqTotTab = pd.crosstab(index=df[predictorColName], columns=df["target"]).copy()
    print(predictorTargetFreqTotTab)
    predictorTargetFreqTotTab.plot(kind="hist", figsize=(8, 8), legend=True,
                                   title="Frequency Plot for "+predictorColName)
    plt.savefig(plots_path + predictorColName + "_" + "TargetFreqProbPlot.png")

def descriptive_stats_and_plots(pa_df, plots_path):
    # See how the data looks like
    pa_df.head(5)
    # pa_data_df_train = pa_data_df_train.drop(pa_data_df_train.index[0])
    pa_df.head(5)
    type(pa_df)
    # From domain knowledge we some columns like userid, doctorid, transdate can be dropped as they would not have
    # any impact on the outcome
    pa_df = pa_df.drop("userid", 1)
    pa_df = pa_df.drop("doctorid", 1)
    pa_df = pa_df.drop("transdate", 1)
    pa_df.shape
    # Convert to appropriate data types
    for col in pa_df.columns:
        pa_df[col] = pa_df[col].astype('category')

    # Print summary statistics
    describe_df = pa_df.describe()
    describe_df
    ax = describe_df[1:2].plot(kind='bar', figsize=(8, 8), legend=True, title="Levels in the categorical attributes")
    ax.set_xlabel("Levels in Attributes")
    ax.set_ylabel("Count")
    fig = ax.get_figure()
    fig.savefig(plots_path + 'LevelsCount.png')
    # Look at the count of the target variable in the dataset
    my_tab_target = pd.crosstab(index=pa_df["target"],  # Make a crosstab
                                columns="count", )  # Name the count column
    # plt.show(block=True)
    my_tab_target.plot(kind="bar", figsize=(8, 8), stacked=True, legend=True)
    plt.savefig(plots_path + 'DisributionOfTarget.png')
    my_tab_target.plot(kind="pie", figsize=(8, 8), stacked=True, legend=True, subplots=True, autopct='%1.1f%%')
    plt.savefig(plots_path + 'PieDisributionOfTarget.png')
    # Look at the relationship between drugclass and target with a 2 way frequency table
    tmp_pa_df = pa_df.copy()
    doPredictorTagetFrequencyPlot("drug", tmp_pa_df, plots_path)
    doPredictorTagetFrequencyPlot("drugclass", tmp_pa_df, plots_path)
    doPredictorTagetFrequencyPlot("druggroup", tmp_pa_df, plots_path)
    doPredictorTagetFrequencyPlot("state", tmp_pa_df, plots_path)
    doPredictorTagetFrequencyPlot("pcn", tmp_pa_df, plots_path)
    doPredictorTagetFrequencyPlot("rxgroupid", tmp_pa_df, plots_path)
    doPredictorTagetFrequencyPlot("bin", tmp_pa_df, plots_path)
    doPredictorTagetFrequencyPlot("ndc", tmp_pa_df, plots_path)
    doPredictorTagetFrequencyPlot("gpi", tmp_pa_df, plots_path)
    doPredictorTagetFrequencyTotalPlot("drug", tmp_pa_df, plots_path)
    doPredictorTagetFrequencyTotalPlot("drugclass", tmp_pa_df, plots_path)
    doPredictorTagetFrequencyTotalPlot("druggroup", tmp_pa_df, plots_path)
    doPredictorTagetFrequencyTotalPlot("state", tmp_pa_df, plots_path)
    doPredictorTagetFrequencyTotalPlot("pcn", tmp_pa_df, plots_path)
    doPredictorTagetFrequencyTotalPlot("rxgroupid", tmp_pa_df, plots_path)
    doPredictorTagetFrequencyTotalPlot("bin", tmp_pa_df, plots_path)
    doPredictorTagetFrequencyTotalPlot("ndc", tmp_pa_df, plots_path)

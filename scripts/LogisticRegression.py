"""
Created on Mon Aug  25 08:43:35 2017

@author: kisho
"""
import statsmodels.api as sm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif,chi2

from scripts.util import dummify


def do_logistic_regression(df, targetColName):
    print("Do logistic regression..")
    print("Dummify all the categorical attributes")
    toZeroOrOne = lambda x: (1 if x == 'TRUE' else 0)
    df[targetColName] = df[targetColName].apply(toZeroOrOne)
    df = dummify(df, targetColName)
    # manually add the intercept
    df['intercept'] = 1.0
    print(df.describe())
    print(df.columns)
    # Create and fit selector
    selector = SelectKBest(f_classif, k=6)
    # Get idxs of columns to keep
    idxs_selected = selector.get_support(indices=True)
    print("Index is ::",idxs_selected)
    features_dataframe_new = feature_data_frame.iloc[:,idxs_selected]
    # Create new dataframe with only desired columns, or overwrite existing
    # features_dataframe_new = feature_data_frame[idxs_selected]
    print(features_dataframe_new.columns)
    logit = sm.Logit(df[targetColName], features_dataframe_new)

    # fit the model
    result = logit.fit()
    print(result.summary())




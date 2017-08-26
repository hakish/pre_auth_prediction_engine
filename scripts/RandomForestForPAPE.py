"""
Created on Mon Aug  25 08:43:35 2017

@author: kisho
"""
from sklearn.ensemble import RandomForestClassifier
from scripts.util import dummify

def create_random_forest_model(df, targetColName):

    print("Create Random Forest Model")
    feature_data_frame = df.drop(targetColName, axis=1)
    target = df[targetColName]
    ## Training!
    rf = RandomForestClassifier(n_estimators=100)  # initialize
    rf.fit(feature_data_frame, target)  # fit the data to the algorithm
    # note - you might get an warning saying you entered a 2 column
    # vector..ignore it. If you know how to get around this warning,
    # please comment! The algorithm seems to work anyway.
    return rf

"""
Created on Mon Aug  25 08:43:35 2017

@author: kisho
"""
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def build_random_forest_model(df, targetColName):

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

def build_neural_network_model(df, targetColName):
    print("Build Neural Net Model")
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (5, 2), random_state = 1)
    feature_data_frame = df.drop(targetColName, axis=1)
    target = df[targetColName]
    clf.fit(feature_data_frame, target)
    return clf

def build_svm_model(df, targetColName):
    print("Build SVM Model")
    clf = svm.SVC()
    feature_data_frame = df.drop(targetColName, axis=1)
    target = df[targetColName]
    clf.fit(feature_data_frame, target)
    return clf
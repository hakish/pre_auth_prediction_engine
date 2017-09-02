"""
Created on Mon Aug  25 08:43:35 2017

@author: kisho
"""
import pickle
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB

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
    filename = '../models/preprocess1/model_random_forest.sav'
    pickle.dump(rf, open(filename, 'wb'))
    return rf

def build_neural_network_model(df, targetColName):
    print("Build Neural Net Model")
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (5, 2), random_state = 1)
    feature_data_frame = df.drop(targetColName, axis=1)
    target = df[targetColName]
    clf.fit(feature_data_frame, target)
    filename = '../models/preprocess1/model_neural_net.sav'
    pickle.dump(clf, open(filename, 'wb'))
    return clf

def build_svm_model(df, targetColName):
    print("Build SVM Model")
    clf = svm.SVC()
    feature_data_frame = df.drop(targetColName, axis=1)
    target = df[targetColName]
    clf.fit(feature_data_frame, target)
    filename = '../models/preprocess1/model_svm.sav'
    pickle.dump(clf, open(filename, 'wb'))
    return clf

def build_nb_model(df, targetColName):
    print("Build Naives Bayes Model")
    mnb = MultinomialNB()
    feature_data_frame = df.drop(targetColName, axis=1)
    target = df[targetColName]
    mnb.fit(feature_data_frame, target)
    filename = '../models/preprocess1/model_naive_bayes.sav'
    pickle.dump(mnb, open(filename, 'wb'))
    return mnb
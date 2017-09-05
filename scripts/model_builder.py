"""
Created on Mon Aug  25 08:43:35 2017

@author: kisho
"""
import pickle
import logging
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

def build_random_forest_model(df, targetColName, model_file_path):

    logging.info("Create Random Forest Model")
    feature_data_frame = df.drop(targetColName, axis=1)
    target = df[targetColName]
    ## Training!
    rf = RandomForestClassifier(n_estimators=100, random_state=42)  # initialize
    rf.fit(feature_data_frame, target)  # fit the data to the algorithm
    # note - you might get an warning saying you entered a 2 column
    # vector..ignore it. If you know how to get around this warning,
    # please comment! The algorithm seems to work anyway.
    scores = cross_val_score(rf, feature_data_frame, target, cv=10)
    logging.info("Scores after performing cross validation :: "+str(scores))
    pickle.dump(rf, open(model_file_path, 'wb'))
    return rf

def build_neural_network_model(df, targetColName, model_file_path):
    logging.info("Build Neural Net Model")
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (5, 2), random_state = 1)
    feature_data_frame = df.drop(targetColName, axis=1)
    target = df[targetColName]
    clf.fit(feature_data_frame, target)
    pickle.dump(clf, open(model_file_path, 'wb'))
    return clf

def build_svm_model(df, targetColName, model_file_path):
    logging.info("Build SVM Model")
    clf = svm.SVC(probability=True)
    feature_data_frame = df.drop(targetColName, axis=1)
    target = df[targetColName]
    clf.fit(feature_data_frame, target)
    pickle.dump(clf, open(model_file_path, 'wb'))
    return clf

def build_nb_model(df, targetColName, model_file_path):
    logging.info("Build Naives Bayes Model")
    mnb = MultinomialNB()
    feature_data_frame = df.drop(targetColName, axis=1)
    target = df[targetColName]
    mnb.fit(feature_data_frame, target)
    pickle.dump(mnb, open(model_file_path, 'wb'))
    return mnb
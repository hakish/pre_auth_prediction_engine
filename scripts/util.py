"""
Created on Mon Aug  7 08:43:35 2017

@author: kisho
"""
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, precision_recall_curve
from sklearn.metrics import confusion_matrix

# Function definitions

def read_csv_data(filename, columnNames):
    df = pd.read_csv(filename, names=columnNames)
    # Get the dimensions of the dataframe
    df.shape
    return df

def drop_cols(df, columns_to_drop):
    df.drop(columns_to_drop, axis=1, inplace=True)
    return df

def dummify(df, targetColName):
    df_no_target = df.copy()
    df_no_target.drop(targetColName, axis=1, inplace=True)
    df_dummied = pd.get_dummies(df_no_target, df_no_target.columns)
    print(df_dummied.head())
    #     We need only dummified columns and as all our data is categorical we only need to consider the dummified df
    df = df_dummied.join(pd.Series.to_frame(df[targetColName]))
    return df

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def create_plot_confusion_matrix(actual, predicted, plots_path, suffix):
    cnf_matrix = confusion_matrix(actual, predicted)
    print("Confusion Matrix for model is :: ", cnf_matrix)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['FALSE', 'TRUE'],
                          title='Confusion matrix, without normalization')
    plt.savefig(str(plots_path)+'CM_Non_Normalized_'+str(suffix)+'.png')
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['FALSE', 'TRUE'], normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig(str(plots_path)+'CM_Normalized_'+str(suffix)+'.png')

def check_col_distribution_and_plot(col, plots_path, colname, suffix):
    y_test_tab = pd.crosstab(index=col,  # Make a crosstab
                             columns="count", )  # Name the count column
    y_test_tab.plot(kind="bar", figsize=(8, 8), stacked=True, legend=True)
    plt.savefig(str(plots_path)+'DisributionOf_'+str(colname)+'_Data_'+str(suffix)+'.png')

def calc_perf_metrics_for_model(model, modelname, actual, predictions, targetcolname, plots_path, suffix):
    # Look at the count of the target variable in the dataset
    check_col_distribution_and_plot(actual, plots_path, targetcolname, suffix)
    create_plot_confusion_matrix(actual, predictions, str(plots_path+modelname+"\\"), suffix)
    print("================ Evaluation metrics for model "+modelname+" [Dataset is "+suffix+"]=======================")
    print("Accuracy of model is ::")
    print(accuracy_score(actual, predictions))
    print("Recall of model is ::")
    print(recall_score(actual, predictions))
    print("Precision score of model is ::")
    print(precision_score(actual, predictions))
    print("F1 score of model is ::")
    print(f1_score(actual, predictions))
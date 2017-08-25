"""
Created on Mon Aug  7 08:43:35 2017

@author: kisho
"""
import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
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

def create_plot_confusion_matrix(actual, predicted, plots_path):
    cnf_matrix = confusion_matrix(actual, predicted)
    print("Confusion Matrix for Random Forest model is :: ", cnf_matrix)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['FALSE', 'TRUE'],
                          title='Confusion matrix, without normalization')
    plt.savefig(plots_path + 'CM_Non_Normalized.png')
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['FALSE', 'TRUE'], normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig(plots_path + 'CM_Normalized.png')

def check_col_distribution_and_plot(col, plots_path):
    y_test_tab = pd.crosstab(index=col,  # Make a crosstab
                             columns="count", )  # Name the count column
    y_test_tab.plot(kind="bar", figsize=(8, 8), stacked=True, legend=True)
    plt.savefig(plots_path + 'DisributionOf_'+col+'Data.png')
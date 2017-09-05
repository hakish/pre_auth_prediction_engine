"""
Created on Mon Aug  7 08:43:35 2017

@author: kisho
"""
import itertools
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn import metrics
from ggplot import *


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
    logging.info(df_dummied.head())
    #     We need only dummified columns and as all our data is categorical we only need to consider the dummified df
    df = df_dummied.join(pd.Series.to_frame(df[targetColName]))
    return df


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function logging.infos and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        logging.info("Normalized confusion matrix")
    else:
        logging.info('Confusion matrix, without normalization')

    logging.info(cm)

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
    logging.info("Confusion Matrix for model is :: "+str(cnf_matrix))
    # np.set_logging.infooptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['FALSE', 'TRUE'],
                          title='Confusion matrix, without normalization')
    plt.savefig(str(plots_path) + 'CM_Non_Normalized_' + str(suffix) + '.png')
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['FALSE', 'TRUE'], normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig(str(plots_path) + 'CM_Normalized_' + str(suffix) + '.png')
    return cnf_matrix


def check_col_distribution_and_plot(col, plots_path, colname, suffix):
    y_test_tab = pd.crosstab(index=col,  # Make a crosstab
                             columns="count", )  # Name the count column
    y_test_tab.plot(kind="bar", figsize=(8, 8), stacked=True, legend=True)
    plt.savefig(str(plots_path) + 'DisributionOf_' + str(colname) + '_Data_' + str(suffix) + '.png')


def calc_perf_metrics_for_model(modelname, actual, predictions, targetcolname, plots_path, suffix):
    # Look at the count of the target variable in the dataset
    check_col_distribution_and_plot(actual, plots_path, targetcolname, suffix)
    cm1 = create_plot_confusion_matrix(actual, predictions, str(plots_path + modelname + "\\"), suffix)
    logging.info(
        "================ Evaluation metrics for model " + modelname + " [Dataset is " + suffix + "]=======================")
    logging.info("Accuracy of model is ::")
    logging.info(accuracy_score(actual, predictions))
    logging.info("Recall of model is ::")
    logging.info(recall_score(actual, predictions))
    logging.info("Precision score of model is ::")
    logging.info(precision_score(actual, predictions))
    logging.info("F1 score of model is ::")
    logging.info(f1_score(actual, predictions))
    sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    print('Sensitivity : ', sensitivity1)
    specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    print('Specificity : ', specificity1)


def do_roc_analysis(actual, pred_probabs, modelname, plots_path, suffix):
    logging.info("Do ROC Analysis..")
    plt.gcf().clear()
    # Compute ROC curve and ROC area for each class
    # fpr, tpr, _ = roc_curve(actual, pred_probabs)
    # logging.info("FPR :: "+str(fpr))
    # logging.info("TPR :: "+str(tpr))
    # # df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
    # # ggplot(df, aes(x='fpr', y='tpr')) + geom_line() + geom_abline(linetype='dashed', slope=1,intercept=0)
    # # auc = metrics.auc(fpr, tpr)
    # # ggplot(df, aes(x='fpr', ymin=0, ymax='tpr')) + geom_area(alpha=0.2) + geom_line(aes(y='tpr')) + ggtitle(
    # #     "ROC Curve w/ AUC=%s" % str(auc))
    # plt.scatter(fpr, tpr)
    # plt.show()
    # auc = np.trapz(tpr, fpr)
    fpr, tpr, threshold = metrics.roc_curve(actual, pred_probabs)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(str(plots_path) + 'ROC_Analysis_' + modelname + '_' + suffix + '.png')
    # plt.show()
    plt.gcf().clear()

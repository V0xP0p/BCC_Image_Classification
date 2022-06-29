# utils.py contains function and classes used in preprocessing, modelling and visualization of
# machine learning models.

import numpy as np
from os.path import exists
import inspect
# import pyyaml
from openpyxl import load_workbook, Workbook
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score


def initialize_function(func_name, *args, **kwargs):
    """
    Call a function with specific parameters initialized otherwise
    it uses the defaults for that function.

    :param func_name: <string> the function we want to call.
    :param kwargs: named arguments consistent with the function called and initialized.
    :return: the function initialized with the new arguments.
    """
    method_params = {}

    func = eval(func_name)

    for parameter in inspect.signature(func).parameters:

        if parameter in kwargs:
            param_value = kwargs.get(parameter)
            method_params[parameter] = param_value

    return func(*args, **method_params)


def split_data(data, labels, split_strategy='StratifiedKFold', **kwargs):
    """
    Splits the data to train-test sets depending on the specified strategy

    :param data: original unsplit data
    :param labels: original unsplit labels
    :param split_strategy: strategy to be followed for the split,
            currently supports: ['StratifiedKFold', 'train_test_split']
    :param kwargs: named arguments

    :return: depending on the method either an array containing the different splits or
            the X_train, X_test, Y_train, Y_test arrays.
    """

    if split_strategy == 'StratifiedKFold':
        split_method = initialize_function(split_strategy, **kwargs)
        return zip(split_method.split(data, labels), range(split_method.n_splits))
    elif split_strategy == 'train_test_split':
        return initialize_function(split_strategy, data, labels, **kwargs)


def get_classifier():
    pass


def compute_roc_auc(index, X, y, clf):
    y_predict = clf.predict_proba(X.iloc[index])[:, 1]
    fpr, tpr, thresholds = roc_curve(y.iloc[index], y_predict)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score


def get_metrics(y_true, y_pred, index=None, data=None, labels=None, classifier=None, verbose=True, filename=None):

    roc_flag = (index is not None) and (data is not None) and (classifier is not None) and (labels is not None)

    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted')
    bacc = balanced_accuracy_score(y_true, y_pred)
    if roc_flag:
        _, _, roc = compute_roc_auc(index, data, labels, classifier)
    else:
        pass
        # raise ValueError(f'For ROC score to be evaluated index,
        # data and classifier fields must be explicitly passed.\n'
        #                  f'Current values:\n'
        #                  f'index= {index}\n'
        #                  f'data= {data}\n'
        #                  f'classifier= {classifier}')
    acc = accuracy_score(y_true, y_pred)

    if verbose:
        print(f"Precision= {precision}")
        print(f"Recall= {recall}")
        print(f"F1 score= {f1}")
        print(f"Balanced Accuracy {bacc}")
        if roc_flag:
            print(f"ROC score {roc}")
        print(f"Accuracy {acc}")

    if filename is not None:
        print(f"Filename is {filename}")
        if exists(filename):
            wb = load_workbook(filename)
            ws = wb.active
            new_row_index = ws.max_row
            row = [f"Experiment {new_row_index}", precision, recall, f1, acc, bacc, roc]
            ws.append(row)
            wb.save(filename)

        else:
            wb = Workbook()
            ws = wb.active

            rows = [[f"Experiment Number", "Precision", "Recall", "F1 score", "Accuracy",
                     "Balanced Accuracy", "ROC-AUC Score"],
                    [f"Experiment 1", precision, recall, f1, acc, bacc, roc]]
            for row in rows:
                ws.append(row)
            wb.save(filename)

    return precision, recall, f1, acc, bacc


if __name__ == '__main__':
    y_true = [0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0]

    get_metrics(y_true, y_pred, verbose=False, filename='results.xlsx')

else:
    print('Main module did not execute')


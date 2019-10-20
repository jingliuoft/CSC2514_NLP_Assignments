from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif, chi2
import numpy as np
from scipy import stats
import argparse
import sys
import os
import csv


def create_model(i):
    if i == 1:
        return SVC(kernel='linear', max_iter=1000)
    elif i == 2:
        return SVC(kernel='rbf', gamma=2, max_iter=1000)
    elif i == 3:
        return RandomForestClassifier(max_depth=5, n_estimators=10)
    elif i == 4:
        return MLPClassifier(alpha=0.05)
    elif i == 5:
        return AdaBoostClassifier()


def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    acc = np.sum(C[i, i] for i in range(len(C))) / np.sum(C)
    return acc


def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    rec = []
    for i in range(len(C)):
        rec.append(C[i][i] / np.sum(C[i]))
    return np.nan_to_num(rec)


def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    prec = []
    for i in range(len(C)):
        prec.append(C[i][i] / np.sum(C[:, i]))
    return np.nan_to_num(prec)


def class31(filename):
    ''' This function performs experiment 3.1


    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:      
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''
    data = np.load(filename)
    lst = data.files
    for item in lst:
        arr = data[item]
    X_train, X_test, y_train, y_test = train_test_split(arr[:, :-1], arr[:, -1], test_size=0.2)
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    result = np.zeros((5, 26))

    for i in range(5):
        model = create_model(i+1)
        model.fit(X_train, y_train)
        predict = model.predict(X_test)
        conf_max = confusion_matrix(y_test, predict)

        result[i, 0] = i+1
        result[i, 1] = accuracy(conf_max)
        result[i, 2:6] = recall(conf_max)
        result[i, 6:10] = precision(conf_max)
        result[i, 10:] = conf_max.reshape((16,))

    iBest = int(result[result[:, 1].argmax(), 0])

    np.savetxt('a1_3.1.csv', result, delimiter=',')

    return (X_train, X_test, y_train, y_test, iBest)


def class32(X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2

    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    result = []

    for size in [1000, 5000, 100000, 15000, 20000]:
        idx = np.random.randint(0, 32000, size=size)

        if size == 1000:
            X_1k = X_train[idx, :]
            y_1k = y_train[idx]

        model = create_model(iBest)
        model.fit(X_train[idx, :], y_train[idx])
        predict = model.predict(X_test)
        conf_max = confusion_matrix(y_test, predict)
        result.append(accuracy(conf_max))

    with open('a1_3.2.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        csv_writer.writerow(result)
        csv_writer.writerow([
            "As the number of training samples increase, the accuracy increases, the accuracy increment becomes slower."
            " The accuracy increment is due to more samples provides more data for classifiers to learn the underneath"
            " trend. As the number of samples increase to a certain amount, existing data could sufficiently express the"
            " trend and the addition of sample size won't provide much \"insight\" to classifier. Therefore the"
            " increment becomes slower as the training samples increase. However, apart from the expected trend, one"
            " unexpected trend is that when the training samples increase and pushing the accuracy to relatively high,"
            " the accuracy slightly drops. One hypothesis could be the introduction of more training samples increases"
            " the variance of the dataset."
        ])

    return (X_1k, y_1k)


def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3

    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    with open('a1_3.3.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)

        # 3.1
        for index, k in enumerate([5, 10, 20, 30, 40, 50]):
            selector = SelectKBest(f_classif, k=k)
            selector.fit_transform(X_train, y_train)
            pp = selector.pvalues_
            result = np.concatenate(([k], pp[selector.get_support()]))
            csv_writer.writerow(result)

        # 3.2
        selector_1k = SelectKBest(f_classif, k=5)
        selector_32k = SelectKBest(f_classif, k=5)

        X_1k_selected = selector_1k.fit_transform(X_1k, y_1k)
        X_1k_test_selected = X_test[:, selector_1k.get_support()]

        X_32k_selected = selector_32k.fit_transform(X_train, y_train)
        X_32k_test_selected = X_test[:, selector_32k.get_support()]

        clf_1k = create_model(i)
        clf_1k.fit(X_1k_selected, y_1k)
        clf_1k_pred = clf_1k.predict(X_1k_test_selected)
        matx_1k = confusion_matrix(y_test, clf_1k_pred)
        acc_1k = accuracy(matx_1k)

        clf_32k = create_model(i)
        clf_32k.fit(X_32k_selected, y_train)
        clf_32k_pred = clf_1k.predict(X_32k_test_selected)
        matx_32k = confusion_matrix(y_test, clf_32k_pred)
        acc_32k = accuracy(matx_32k)

        csv_writer.writerow([acc_1k, acc_32k])
        csv_writer.writerow([
            "For low (1k) and higher (32k) amount of input data, the common significant features when the selector is"
            " set for best 5 features are \"receptiviti_ambitious\" and \"receptiviti_intellectual\". The two extracted"
            " features has high political direction. And therefore a good feature for the political leaning. *note as"
            " the selected features are highly related to the data sampled in 3.1, multiple runs have been conducted."
        ])
        csv_writer.writerow([
            "The p-values are generally lower given more data. One of the possible explanation is that when more data"
            " is included in training, there are more variance in each of the features, and therefore some of those"
            " features becomes a valuable feature for prediction."
        ])
        csv_writer.writerow([
            "The 5 features for the 32k training case are: \"receptiviti_ambitious\", \"receptiviti_conscientiousness\","
            " \"receptiviti_intellectual\", \"receptiviti_power_driven\", and \"receptiviti_work_oriented\".  All the 5"
            " features are highly related to work and politics as \"ambitious\", \"conscientiousness\","
            " \"power_driven\" and \"work_oriented\" describe the personality and \"intellectual\" describes the"
            " competency of the user."
        ])


def class34(filename, i):
    ''' This function performs experiment 3.4

    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    data = np.load(filename)
    lst = data.files
    for item in lst:
        arr = data[item]

    kf = KFold(n_splits=5, shuffle=True)
    result = np.empty((5, 5))
    n = 0

    for train_index, test_index in kf.split(arr):
        X_train_fold, X_test_fold = arr[train_index, :-1], arr[test_index, :-1]
        y_train_fold, y_test_fold = arr[train_index, -1], arr[test_index, -1]

        for index in range(5):
            model = create_model(index + 1)
            model.fit(X_train_fold, y_train_fold)
            predict = model.predict(X_test_fold)
            conf_max = confusion_matrix(y_test_fold, predict)
            result[n, index] = accuracy(conf_max)

        n += 1

    accuracy_compare = []
    for m in range(5):
        if m != i-1:
            accuracy_compare.append(stats.ttest_rel(result[:, i - 1], result[:, m]).pvalue)

    with open('a1_3.4.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        csv_writer.writerows(result)
        csv_writer.writerow(accuracy_compare)
        csv_writer.writerow([
            "Based on the accuracy result, AdaBoost outperforms the rest 4 models. One of the possible explanations is"
            " that AdaBoost is designed to convert a set of weak classifiers into a strong one. It focuses on"
            " misclassified data to improve accuracy. For the Raddit dataset, or general nlp related applications,"
            " there is no fixed, significant features for classification. Therefore, AdaBoost is a great fit for the"
            " problem."
        ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()

    # complete each classification experiment, in sequence.
    (X_train, X_test, y_train, y_test, iBest) = class31(args.input)
    (X_1k, y_1k) = class32(X_train, X_test, y_train, y_test, iBest)
    class33(X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    class34(args.input, iBest)

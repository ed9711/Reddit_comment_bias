import argparse
import os
import numpy as np
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
import datetime

start_time = datetime.datetime.now()


def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    if C.sum() != 0:
        return C.trace() / C.sum()
    else:
        return 0.0


def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    classes = []
    C = C.T
    for j in range(C.shape[0]):
        if C[j].sum() != 0:
            classes.append(C[j][j] / C[j].sum())
        else:
            classes.append(0.0)
    return classes


def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    classes = []
    for i in range(C.shape[0]):
        if C[i].sum() != 0:
            classes.append(C[i][i] / C[i].sum())
        else:
            classes.append(0.0)
    return classes


def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:      
       i: int, the index of the supposed best classifier
    '''
    print("Class3.1")
    classifiers = [SGDClassifier(), GaussianNB(),
                   RandomForestClassifier(max_depth=5, n_estimators=10, random_state=401), MLPClassifier(),
                   AdaBoostClassifier(random_state=401)]
    accuracies = {}
    for clf in classifiers:
        print(str(clf).split('(')[0])
        clf.fit(X_train, y_train)
        pred_label = clf.predict(X_test)
        conf_matrix = confusion_matrix(y_test, pred_label)
        acc = accuracy(conf_matrix)
        rec = recall(conf_matrix)
        prec = precision(conf_matrix)
        accuracies[str(clf).split('(')[0]] = [acc, rec, prec, conf_matrix]

    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
        # For each classifier, compute results and write the following output:
        for key in accuracies:
            outf.write(f'Results for {key}:\n')  # Classifier name
            outf.write(f'\tAccuracy: {round(accuracies[key][0], 4)}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in accuracies[key][1]]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in accuracies[key][2]]}\n')
            outf.write(f'\tConfusion Matrix: \n{accuracies[key][3]}\n\n')

    best = 0
    iBest = 999
    for key in accuracies:
        if accuracies[key][0] > best:
            best = accuracies[key][0]
            iBest = list(accuracies.keys()).index(key)
    return iBest


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    amount = [1000, 5000, 10000, 15000, 20000]
    accuracies = []
    print("class3.2")
    for size in amount:
        if iBest == 0:
            clf = SGDClassifier()
        elif iBest == 1:
            clf = GaussianNB()
        elif iBest == 2:
            clf = RandomForestClassifier(max_depth=5, random_state=401, n_estimators=10)
        elif iBest == 3:
            clf = MLPClassifier()
        elif iBest == 4:
            clf = AdaBoostClassifier(random_state=401)
        print(size)
        print(str(clf).split('(')[0])
        clf.fit(X_train[:size], y_train[:size])
        pred_label = clf.predict(X_test)
        conf_matrix = confusion_matrix(y_test, pred_label)
        acc = accuracy(conf_matrix)
        accuracies.append(acc)

    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        # For each number of training examples, compute results and write
        for i in range(len(amount)):
            outf.write(f'{amount[i]}: {round(accuracies[i], 4)}\n')

    X_1k = X_train[:1000]
    y_1k = y_train[:1000]
    return (X_1k, y_1k)


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''

    if i == 0:
        clf = SGDClassifier()
    elif i == 1:
        clf = GaussianNB()
    elif i == 2:
        clf = RandomForestClassifier(max_depth=5, random_state=401, n_estimators=10)
    elif i == 3:
        clf = MLPClassifier()
    elif i == 4:
        clf = AdaBoostClassifier(random_state=401)
    # clf = RandomForestClassifier(max_depth=5, random_state=401, n_estimators=10)
    num_features = [5, 50]
    k_pp = {}

    # 3.3.1 32k
    print("Class33")
    print(str(clf).split('(')[0])
    print("3.3.1")
    for k in num_features:
        selector = SelectKBest(f_classif, k=k)
        X_new = selector.fit_transform(X_train, y_train)
        pp = selector.pvalues_
        k_pp[k] = pp

    # 3.3.2 1k
    print("3.3.2.1k")
    if i == 0:
        clf = SGDClassifier()
    elif i == 1:
        clf = GaussianNB()
    elif i == 2:
        clf = RandomForestClassifier(max_depth=5, random_state=401, n_estimators=10)
    elif i == 3:
        clf = MLPClassifier()
    elif i == 4:
        clf = AdaBoostClassifier(random_state=401)
    selector1 = SelectKBest(f_classif, k=5)
    X_new = selector1.fit_transform(X_1k, y_1k)
    X_new_test = selector1.transform(X_test)
    clf.fit(X_new, y_1k)
    pred_label = clf.predict(X_new_test)
    conf_matrix = confusion_matrix(y_test, pred_label)
    acc_1k = accuracy(conf_matrix)

    # 3.3.2 32k
    print("3.3.2.32k")
    if i == 0:
        clf = SGDClassifier()
    elif i == 1:
        clf = GaussianNB()
    elif i == 2:
        clf = RandomForestClassifier(max_depth=5, random_state=401, n_estimators=10)
    elif i == 3:
        clf = MLPClassifier()
    elif i == 4:
        clf = AdaBoostClassifier(random_state=401)

    selector32 = SelectKBest(f_classif, k=5)
    X_new = selector32.fit_transform(X_train, y_train)
    X_new_test = selector32.transform(X_test)
    clf.fit(X_new, y_train)
    pred_label = clf.predict(X_new_test)
    conf_matrix = confusion_matrix(y_test, pred_label)
    acc_32k = accuracy(conf_matrix)

    # 3.3.3
    print("3.3.3")
    pp32 = np.array(selector32.pvalues_)
    pp1 = np.array(selector1.pvalues_)
    count = pp32.tolist().count(0)
    if count < 5:
        count = 5
    idx32 = np.argpartition(pp32, count)
    idx1 = np.argpartition(pp1, 5)
    intersect = np.intersect1d(idx32[:count], idx1[:5])

    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # Prepare the variables with corresponding names, then uncomment
        # this, so it writes them to outf.

        for k in num_features:
            outf.write(f'{k} p-values: {[round(pval, 4) for pval in k_pp[k]]}\n')

        outf.write(f'Accuracy for 1k: {acc_1k:.4f}\n')
        outf.write(f'Accuracy for full dataset: {acc_32k:.4f}\n')
        outf.write(f'Chosen feature intersection: {intersect}\n')
        outf.write(f'Top-5 at higher: {idx32[:5]}\n')
        outf.write("My answers:\n")
        outf.write(
            """(a): 12.Number of adverbs, 22. Standard deviation of IMG from Bristol, Gilhooly, and Logie norms, 150.
        receptiviti_intellectual, 164.receptiviti_self_conscious. \n For number 12, the adverbs are modifiers for verb, 
        adjectives. For more extreme political groups, they might be more likely to try to magnify/minimize situation 
        to their advantage, thus using more adverbs.\n For number 22 the Imageability measures how easy a word can trigger
        mental images. This can showcase the difference in age. Younger people have more active imagination and use more
        imagenitive words. Studies have shown that the Left tend to be younger than the Right in general. Thus, this
        feature can be used to differentiate different political groups.\n For number 164, self conscious can be one of the 
        major reasons that show differentiation between political group. People that are more extreme in their ideology 
        can be less self conscious than more center people. This quality is why they can accept more extreme ideologies 
        easier than others. This is also true for extreme groups/ideology in general. Thus lack of self consciousness 
        could be related to political extremeness\n For number 150, being intellectual can be a determinate for education
        level, or pretentiousness. Studies have show that the level of education and class are related to which political
        group one identify with. Thus, it can be useful in filtering the comments into classes.\n
        (b): P-values are generally lower when given more data. This is probably because the more data we have, the 
        more identifiable each feature is to a certain political group.\n
        (c): The only feature in the 32k training case is number 21.Standard deviation of AoA (100-700) from Bristol, 
        Gilhooly, and Logie norms. Age of acquisition refers to the age when a word is learned. This feature can 
        showcase the difference in age and education, both are related to different political affiliations. This 
        feature would become more prominent as the data becomes larger because there are more words, so the STD of AoA 
        becomes more accurate.
        """)


def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    kfold = KFold(n_splits=5, random_state=401, shuffle=True)
    X = np.zeros((X_train.shape[0] + X_test.shape[0], X_train.shape[1]))
    Y = np.zeros((y_train.shape[0] + y_test.shape[0]))
    X[:X_train.shape[0], :] = X_train
    X[X_train.shape[0]:, :] = X_test
    Y[:y_train.shape[0]] = y_train
    Y[y_train.shape[0]:] = y_test

    accuracies = {}
    accuracies[0] = []
    accuracies[1] = []
    accuracies[2] = []
    accuracies[3] = []
    accuracies[4] = []
    for train_id, test_id in kfold.split(X):
        train_data, test_data = X[train_id], X[test_id]
        train_label, test_label = Y[train_id], Y[test_id]
        print("new fold")
        for item in range(5):
            if item == 0:
                clf = SGDClassifier()
            elif item == 1:
                clf = GaussianNB()
            elif item == 2:
                clf = RandomForestClassifier(max_depth=5, random_state=401, n_estimators=10)
            elif item == 3:
                clf = MLPClassifier()
            elif item == 4:
                clf = AdaBoostClassifier(random_state=401)

            clf.fit(train_data, train_label)
            pred_label = clf.predict(test_data)
            conf_matrix = confusion_matrix(test_label, pred_label)
            acc = accuracy(conf_matrix)
            accuracies[item].append(acc)

    p_values = []
    for key in accuracies:
        if key != i:
            S = ttest_rel(accuracies[key], accuracies[i])
            print(S.pvalue)
            p_values.append(S.pvalue)

    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        # Prepare kfold_accuracies, then uncomment this, so it writes them to outf.
        for item in range(5):
            outf.write(f'Kfold Accuracies: {[round(item, 4) for item in accuracies[key] for key in accuracies]}\n')
        outf.write(f'p-values: {[round(pval, 4) for pval in p_values]}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()

    # TODO: load data and split into train and test.
    feats = np.load(args.input)
    feats = feats[feats.files[0]]
    # print(feats.shape)
    data = feats[:, :-1]
    labels = feats[:, -1]
    train_data, test_data, train_label, test_label = train_test_split(data, labels, train_size=0.8, random_state=401)
    # TODO : complete each classification experiment, in sequence.
    iBest = class31(args.output_dir, train_data, test_data, train_label, test_label)
    (X_1k, y_1k) = class32(args.output_dir, train_data, test_data, train_label, test_label, iBest)
    class33(args.output_dir, train_data, test_data, train_label, test_label, iBest, X_1k, y_1k)
    class34(args.output_dir, train_data, test_data, train_label, test_label, iBest)
    end_time = datetime.datetime.now()
    print("Start time: {}".format(start_time))
    print("End time: {}".format(end_time))

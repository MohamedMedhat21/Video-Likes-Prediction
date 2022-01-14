import time
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pickle


def logisticRegression(X_train, X_test, Y_train, Y_test):
    t0 = time.time()
    logreg = LogisticRegression().fit(X_train, Y_train)
    t1 = time.time()
    filename = 'LogisticRegression.sav'
    pickle.dump(logreg, open(filename, 'wb'))
    training_Time = (t1 - t0)
    t0 = time.time()
    y_pred = logreg.predict(X_test)
    t1 = time.time()
    testing_Time = (t1 - t0)
    accuracy = accuracy_score(y_pred, Y_test)
    return accuracy, training_Time, testing_Time


def SVMPolykernel(X_train, X_test, Y_train, Y_test, c=2, d=3):
    t0 = time.time()
    clf = svm.SVC(C=c, kernel='poly', degree=d).fit(X_train, Y_train)
    t1 = time.time()
    filename = 'SVMPolynomial.sav'
    pickle.dump(clf, open(filename, 'wb'))
    training_Time = (t1 - t0)
    t0 = time.time()
    y_pred = clf.predict(X_test)
    t1 = time.time()
    testing_Time = (t1 - t0)
    accuracy = accuracy_score(y_pred, Y_test)
    return accuracy, training_Time, testing_Time


def decisionTree(X_train, X_test, Y_train, Y_test):
    t0 = time.time()
    clf = DecisionTreeClassifier().fit(X_train, Y_train)
    t1 = time.time()
    filename = 'decisionTree.sav'
    pickle.dump(clf, open(filename, 'wb'))
    training_Time = (t1 - t0)
    t0 = time.time()
    y_pred = clf.predict(X_test)
    t1 = time.time()
    testing_Time = (t1 - t0)
    accuracy = accuracy_score(y_pred, Y_test)
    return accuracy, training_Time, testing_Time


def SVMRBFkernel(X_train, X_test, Y_train, Y_test, c=3, g=3.1):
    t0 = time.time()
    rbf_svc = svm.SVC(kernel='rbf', gamma=g, C=c).fit(X_train, Y_train)
    t1 = time.time()
    filename = 'SVMRBFkernel.sav'
    pickle.dump(rbf_svc, open(filename, 'wb'))
    training_Time = (t1 - t0)
    t0 = time.time()
    y_pred = rbf_svc.predict(X_test)
    t1 = time.time()
    testing_Time = (t1 - t0)
    accuracy = accuracy_score(y_pred, Y_test)
    return accuracy, training_Time, testing_Time

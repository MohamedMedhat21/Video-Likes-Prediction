import numpy as np

from milestone2.helper_Functions import *
from milestone2.models import *

X_train, X_test, Y_train, Y_test = pre_processing()

# SVM with Polynomial kernel
polyC = [0.5, 0.9, 2]
polyD = 3
polyArr = []
for i in polyC:
    SVMPolykernelAccuracy, training_Time, testing_Time = SVMPolykernel(X_train, X_test, Y_train, Y_test, i, polyD)
    polyArr.append([i, polyD, SVMPolykernelAccuracy * 100])
    print('Training Time Taken by SVM with Polynomial kernel', training_Time, 'seconds')
    print('Testing Time Taken by SVM with Polynomial kernel', testing_Time, 'seconds')
    print('SVM with Polynomial kernel Accuracy', SVMPolykernelAccuracy, ' with C=', i, 'with degree=', polyD)
    print("-----------------------------------------------------------------------------")

polyArr = np.array(polyArr)
polyDF = pd.DataFrame(polyArr, columns=['Regularization parameter(C)', 'Degree', 'Accuracy'])
polyDF.to_csv('PolySVM Tuning.csv')

# SVM with Gaussian(RBF) kernel
rbfC = [0.1, 0.8, 1, 3, 3, 3, 3, 3]
rbfG = [0.8, 0.8, 0.8, 0.8, 1, 2, 3.1, 3.2]
rbfArr = []
for i in range(0, len(rbfG)):
    SVMRBFkernelAccuracy, training_Time, testing_Time = SVMRBFkernel(X_train, X_test, Y_train, Y_test, rbfC[i], rbfG[i])
    rbfArr.append([rbfC[i], rbfG[i], SVMRBFkernelAccuracy * 100])
    print('Training Time Taken by SVM with Gaussian(RBF) kernel', training_Time, 'seconds')
    print('Testing Time Taken by SVM with Gaussian(RBF) kernel', testing_Time, 'seconds')
    print('SVM with Gaussian(RBF) kernel Accuracy', SVMRBFkernelAccuracy, ' with C=', rbfC[i], 'with gamma=', rbfG[i])
    print("-----------------------------------------------------------------------------")

rbfArr=np.array(rbfArr)
rbfDF = pd.DataFrame(rbfArr, columns=['Regularization parameter(C)', 'Variance(gamma)', 'Accuracy'])
rbfDF.to_csv('rbfSVM Tuning.csv')

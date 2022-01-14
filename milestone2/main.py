from milestone2.helper_Functions import *
from milestone2.models import *
from matplotlib import pyplot as plt

X_train, X_test, Y_train, Y_test = pre_processing()
trainingTimes = []
testingTimes = []

# Logistic Regression
logisticRegressionAccuracy, training_Time, testing_Time = logisticRegression(X_train, X_test, Y_train, Y_test)
trainingTimes.append(training_Time)
testingTimes.append(testing_Time)
print("-----------------------------------------------------------------------------")
print('Training Time Taken by Logistic Regression', training_Time, 'seconds')
print('Testing Time Taken by Logistic Regression', testing_Time, 'seconds')
print('Logistic Regression Accuracy', logisticRegressionAccuracy)
print("-----------------------------------------------------------------------------")

# SVM with Polynomial kernel
polyC = 2
polyD = 3
SVMPolykernelAccuracy, training_Time, testing_Time = SVMPolykernel(X_train, X_test, Y_train, Y_test, polyC, polyD)
trainingTimes.append(training_Time)
testingTimes.append(testing_Time)
print('Training Time Taken by SVM with Polynomial kernel', training_Time, 'seconds')
print('Testing Time Taken by SVM with Polynomial kernel', testing_Time, 'seconds')
print('SVM with Polynomial kernel Accuracy', SVMPolykernelAccuracy, ' with C=', polyC, 'with degree=', polyD)
print("-----------------------------------------------------------------------------")

# Decision Tree
decisionTreeAccuracy, training_Time, testing_Time = decisionTree(X_train, X_test, Y_train, Y_test)
trainingTimes.append(training_Time)
testingTimes.append(testing_Time)
print('Training Time Taken by Decision Tree', training_Time, 'seconds')
print('Testing Time Taken by Decision Tree', testing_Time, 'seconds')
print('Decision Tree Accuracy', decisionTreeAccuracy)
print("-----------------------------------------------------------------------------")

# SVM with Gaussian(RBF) kernel
rbfC = 3
rbfG = 3.1
SVMRBFkernelAccuracy, training_Time, testing_Time = SVMRBFkernel(X_train, X_test, Y_train, Y_test, rbfC, rbfG)
trainingTimes.append(training_Time)
testingTimes.append(testing_Time)
print('Training Time Taken by SVM with Gaussian(RBF) kernel', training_Time, 'seconds')
print('Testing Time Taken by SVM with Gaussian(RBF) kernel', testing_Time, 'seconds')
print('SVM with Gaussian(RBF) kernel Accuracy', SVMRBFkernelAccuracy, ' with C=', rbfC, 'with gamma=', rbfG)
print("-----------------------------------------------------------------------------")

modelsNames = ['Logistic Regression', 'SVM(Polynomial)', 'Decision Tree', 'SVM(RBF)']

# classification accuracy bar graph
accuracies = [82, 75.5, 92, 94]
fig = plt.figure(figsize=(5, 5))
fig.suptitle('Classification Accuracy', fontsize=20)
plt.bar(modelsNames, accuracies, width=0.6)
plt.xlabel('Models', fontsize=15)
plt.ylabel('Accuracies', fontsize=15)
plt.yticks(np.arange(0, 105, step=5))
plt.xticks(fontsize=6)
plt.show()

# total training time bar graph
fig = plt.figure(figsize=(5, 5))
fig.suptitle('Total Training Time', fontsize=20)
plt.bar(modelsNames, trainingTimes, width=0.6)
plt.xlabel('Models', fontsize=15)
plt.ylabel('Training Time', fontsize=15)
plt.yticks(np.arange(0, 100, step=5))
plt.xticks(fontsize=6)
plt.show()

# total testing time bar graph
fig = plt.figure(figsize=(5, 5))
fig.suptitle('Total Testing Time', fontsize=20)
plt.bar(modelsNames, testingTimes, width=0.6)
plt.xlabel('Models', fontsize=15)
plt.ylabel('Testing Time', fontsize=15)
plt.ylim(0,16)
plt.yticks(np.arange(0, 16, step=1))
plt.xticks(fontsize=6)
plt.show()

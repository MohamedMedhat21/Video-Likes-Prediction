import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score


def feature_scaling(X, a, b):
    X = np.array(X)
    Normalized_X = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:, i] = ((X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))) * (b - a) + a
    return Normalized_X


def feature_encoder(X):
    lbl = LabelEncoder()
    lbl.fit(list(X.values))
    X = lbl.transform(list(X.values))
    return X


data = pd.read_csv("VideoLikesDatasetClassification.csv")

#data.dropna(how='any', inplace=True)
data["tags"].fillna("[none]", inplace=True)
data['views'].fillna(int((data['views'].mean())), inplace=True)
data['comment_count'].fillna(int((data['comment_count'].mean())), inplace=True)
data['category_id'].fillna((data['category_id'].max()), inplace=True)

data["video_id"] = feature_encoder(data["video_id"])
data["title"] = feature_encoder(data["title"])
data["channel_title"] = feature_encoder(data["channel_title"])
data["comments_disabled"] = feature_encoder(data["comments_disabled"])
data["ratings_disabled"] = feature_encoder(data["ratings_disabled"])
data["video_error_or_removed"] = feature_encoder(data["video_error_or_removed"])
data["tags"] = feature_encoder(data["tags"])
data["VideoPopularity"] = feature_encoder(data["VideoPopularity"])

# Correlation matrix to help us in features selection
corr_matrix = data.corr()
best_features = corr_matrix.index[abs(corr_matrix['VideoPopularity']) > 0.04]
best_features = best_features.delete(-1)
X = data[best_features]
X = feature_scaling(X, 0, 100)
Y = data[['VideoPopularity']].iloc[:, :]
Y = feature_scaling(Y, 0, 100)
Y = Y.flatten()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1)

filename = 'LogisticRegression.sav'
# load the model from disk
LogisticRegression = pickle.load(open(filename, 'rb'))
y_pred = LogisticRegression.predict(X_test)
logisticRegressionAccuracy = accuracy_score(y_pred, y_test)
print('Logistic Regression Accuracy', logisticRegressionAccuracy)
print("-----------------------------------------------------------------------------")

filename = 'SVMPolynomial.sav'
# load the model from disk
SVMPolynomial = pickle.load(open(filename, 'rb'))
y_pred = SVMPolynomial.predict(X_test)
SVMPolykernelAccuracy = accuracy_score(y_pred, y_test)
print('SVM with Polynomial kernel Accuracy', SVMPolykernelAccuracy)
print("-----------------------------------------------------------------------------")

filename = 'decisionTree.sav'
# load the model from disk
decisionTree = pickle.load(open(filename, 'rb'))
y_pred = decisionTree.predict(X_test)
decisionTreeAccuracy = accuracy_score(y_pred, y_test)
print('Decision Tree Accuracy', decisionTreeAccuracy)

print("-----------------------------------------------------------------------------")
filename = 'SVMRBFkernel.sav'
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
y_pred = loaded_model.predict(X_test)
SVMRBFkernelAccuracy = accuracy_score(y_pred, y_test)
print('SVM with Gaussian(RBF) kernel Accuracy', SVMRBFkernelAccuracy)

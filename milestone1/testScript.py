import pandas as pd
import numpy as np
import pickle
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

def feature_scaling(X,a,b):
    X = np.array(X)
    Normalized_X=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:,i]=((X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i])))*(b-a)+a
    return Normalized_X

#Read data
data = pd.read_csv("VideoLikesDataset.csv")

#Handling missing data
data['views'].fillna(int((data['views'].mean())), inplace=True)
data['comment_count'].fillna(int((data['comment_count'].mean())), inplace=True)
data['likes'].fillna(int((data['likes'].mean())), inplace=True)

#Extract features and output
X = data[['views', 'comment_count']].iloc[:,:]
X = feature_scaling(X, 0, 100)
y_test = data[['likes']].iloc[:,:]
y_test = feature_scaling(y_test, 0, 100)

#Predict
filename1 = 'Model1.sav'
loaded_model1 = pickle.load(open(filename1, 'rb'))
prediction1 = loaded_model1.predict(X)
print('Multiple linear regression model values:')
print('Mean Square Error of linear regression model', metrics.mean_squared_error(np.asarray(y_test), prediction1))
print('R2 score', r2_score(np.asarray(y_test), prediction1))

poly_features = PolynomialFeatures(degree=4)
X_poly = poly_features.fit_transform(X)
filename2 = 'Model2.sav'
loaded_model2 = pickle.load(open(filename2, 'rb'))
prediction2 = loaded_model2.predict(X_poly)
print('\nPolynomial regression model values:')
print('Mean Square Error of linear regression model', metrics.mean_squared_error(np.asarray(y_test), prediction2))
print('R2 score', r2_score(np.asarray(y_test), prediction2))

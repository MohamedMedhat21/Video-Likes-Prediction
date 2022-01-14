from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, PolynomialFeatures
from sklearn import metrics
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import pickle

def feature_scaling(X,a,b):
    X = np.array(X)
    Normalized_X=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:,i]=((X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i])))*(b-a)+a
    return Normalized_X

def feature_encoder(X):
    lbl = LabelEncoder()
    lbl.fit(list(X.values))
    X = lbl.transform(list(X.values))
    return X


print('-----> Model 2 - Milestone 1:\n')
#Read data
data = pd.read_csv("VideoLikesDataset.csv")

#Drop rows of blank values
data.dropna(how='any', inplace=True)

#Drop removed videos
data = data[data['video_error_or_removed'] == False]

#Drop Videos with no name id -> #NAMES?
data = data[data['video_id'] != "#NAME?"]

#Handle date-time format
data['trending_date'] = pd.to_datetime(data['trending_date'], format='%y.%d.%m')
data['publish_time'] = pd.to_datetime(data['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')

#Split date into 2 columns
data.insert(5, 'publish_date', data['publish_time'].dt.date)
data['publish_time'] = data['publish_time'].dt.time
data['publish_date'] = pd.to_datetime(data['publish_date'])

#Encode features
data["video_id"] = feature_encoder(data["video_id"])
data["title"] = feature_encoder(data["title"])
data["channel_title"] = feature_encoder(data["channel_title"])
data["comments_disabled"] = feature_encoder(data["comments_disabled"])
data["ratings_disabled"] = feature_encoder(data["ratings_disabled"])
data["video_error_or_removed"] = feature_encoder(data["video_error_or_removed"])
data["tags"] = feature_encoder(data["tags"])

#Correlation matrix to help us in features selection
columns_of_interest = ['views', 'likes','comment_count', 'channel_title', 'category_id']
corr_matrix = data[columns_of_interest].corr()
print("Correlation Matrix:\n", corr_matrix)
print("-----------------------------------------------------------------------------\n\n")

#Get features with more than 50% correlation with likes using heatmap
corr = data.corr()
best_features = corr.index[abs(corr['likes']>0.5)]
plt.subplots(figsize=(6, 4))
top_corr = data[best_features].corr()
sns.heatmap(top_corr, annot=True)
plt.show()

#Extract features and output
X = data[['views', 'comment_count']].iloc[:,:]
X = feature_scaling(X, 0, 100)
Y = data[['likes']].iloc[:,:]
Y = feature_scaling(Y, 0, 100)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=1)

#Polynomial regression
t2 = time.time()
poly_features = PolynomialFeatures(degree=4)
X_train_poly = poly_features.fit_transform(X_train)
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)
filename = 'Model2.sav'
pickle.dump(poly_model, open(filename, 'wb'))
prediction = poly_model.predict(poly_features.fit_transform(X_test))
t3 = time.time()
print('\nPolynomial regression model values:')
print('Co-efficient of polynomial regression',poly_model.coef_)
print('Intercept of polynomial regression model',poly_model.intercept_)
print('Mean Square Error of polynomial regression model', metrics.mean_squared_error(y_test, prediction))
print('R2 score', r2_score(np.asarray(y_test), prediction))
print('Time taken', 1000*(t3 - t2))
print("-----------------------------------------------------------------------------")
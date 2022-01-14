import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns


def pre_processing():
    # Read data
    data = pd.read_csv("VideoLikesDatasetClassification.csv")

    # Drop rows of blank values
    data.dropna(how='any', inplace=True)

    # Encode features
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

    # print("Correlation Matrix:\n", corr_matrix)
    # print("-----------------------------------------------------------------------------")

    corr_matrix.to_csv('Correlation Matrix.csv')

    # plt.subplots(figsize=(6, 4))
    # top_corr = data[best_features].corr()
    # sns.heatmap(top_corr, annot=True)
    # plt.show()

    # X = data[['views', 'comment_count']].iloc[:, :]
    X = feature_scaling(X, 0, 100)
    Y = data[['VideoPopularity']].iloc[:, :]
    Y = feature_scaling(Y, 0, 100)
    Y = Y.flatten()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1)

    return X_train, X_test, y_train, y_test


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

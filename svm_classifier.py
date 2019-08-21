import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from sklearn import svm, datasets

poems = pd.read_csv('poems.csv')
english_poems = poems.loc[poems.country.isin(['American', 'English'])]
mask = (english_poems.groupby('author')['author'].transform(len) > 2)
english_poems = english_poems[mask]

train_idx, test_idx = train_test_split(english_poems.index, test_size=0.2, random_state=4, stratify=english_poems.author)

poems_train = english_poems.body.loc[train_idx]
poems_test = english_poems.body.loc[test_idx]

vectorizer = TfidfVectorizer()
vectorizer = vectorizer.fit(poems_train.values)

features_train = vectorizer.transform(poems_train.values)
features_test = vectorizer.transform(poems_test.values)

author_train = english_poems.author.loc[train_idx]
author_test = english_poems.author.loc[test_idx]


C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C, max_iter=10000),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))
models = [clf.fit(features_train, author_train) for clf in models]

print('SVC with linear kernel')
predictions = models[0].predict(features_test)
print("Test set Accuracy: ", accuracy_score(author_test, predictions))

print('LinearSVC (linear kernel)')
predictions = models[1].predict(features_test)
print("Test set Accuracy: ", accuracy_score(author_test, predictions))

print('SVC with RBF kernel')
predictions = models[2].predict(features_test)
print("Test set Accuracy: ", accuracy_score(author_test, predictions))

print('SVC with polynomial (degree 3) kernel')
predictions = models[3].predict(features_test)
print("Test set Accuracy: ", accuracy_score(author_test, predictions))

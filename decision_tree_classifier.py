import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import tree
from sklearn.metrics import accuracy_score

poems = pd.read_csv('poems.csv')
english_poems = poems.loc[poems.country.isin(['American', 'English'])]
mask = (english_poems.groupby('author')['author'].transform(len) > 10)
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

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, author_train)
predictions = clf.predict(features_test)

print("Test set Accuracy: ", accuracy_score(author_test, predictions))

# viewing incorrect predictions
error_mask = predictions != author_test
error_idx = test_idx[error_mask]
error_poems = english_poems.loc[error_idx]

# actual, predicted
mistakes = list(zip(error_poems["author"], predictions[error_mask]))
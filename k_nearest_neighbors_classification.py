import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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

Ks = 20
mean_acc = np.zeros((Ks - 1))
std_acc = np.zeros((Ks - 1))
ConfustionMx = []

for n in range(1, Ks):
    neigh = KNeighborsClassifier(n_neighbors = n).fit(features_train, author_train)
    predictions = neigh.predict(features_test)
    print(f'Test set Accuracy for {n} neighors: {accuracy_score(author_test, predictions)}')
    mean_acc[n - 1] = accuracy_score(author_test, predictions)
    std_acc[n - 1] = np.std(predictions == author_test) / np.sqrt(predictions.shape[0])
mean_acc

plt.plot(range(1,Ks),mean_acc, 'g')
plt.fill_between(range(1, Ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

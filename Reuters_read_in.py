import numpy as np
import pandas as pd
from sklearn.datasets import fetch_rcv1, fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import sqlite3


# 20 newsgroups - Train
politics = ['talk.politics.mideast', 'talk.politics.guns']
newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'), categories=politics)
vectorizer = TfidfVectorizer()

# Take just 10 examples as active learning
vectors = vectorizer.fit_transform(newsgroups_train.data)[0:20]
clf = LogisticRegression(C=10)
clf.fit(vectors, newsgroups_train.target[0:20])

# Test Set
newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'), categories=politics)
vectors_test = vectorizer.transform(newsgroups_test.data)
metrics.f1_score(newsgroups_test.target, clf.predict(vectors_test), average='binary')
test_set = pd.DataFrame(newsgroups_test.data)
test_set['class_proba'] = clf.predict_proba(vectors_test)[:,1] # Probability of being in class 1, which is the second column returned by .predict_proba
test_set['prediction'] = clf.predict(vectors_test)
test_set.columns = ['text', 'class_proba', 'prediction']
test_set['predicted_labels'] = test_set['prediction'].map({0:'Middle-east', 1:'Guns'})

conn = sqlite3.connect('reuters_testX_with_proba.sqlite')
c = conn.cursor()
test_set.to_sql('reuters_test_X', conn)
conn.commit()
conn.close()

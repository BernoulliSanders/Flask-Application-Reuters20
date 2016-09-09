import numpy as np
import pandas as pd
from sklearn.datasets import fetch_rcv1, fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import sqlite3

'''
rcv1 = fetch_rcv1()

#pd.read_pickle('lyrl2004_tokens_train.dat')


f = open('lyrl2004_tokens_train.dat', 'rb')  # We need to re-open the file
data = f.read()
train = np.fromfile(data,sep="")
f.close()'''

# 20 newsgroups
politics = ['talk.politics.mideast', 'talk.politics.guns']
newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'), categories=politics)
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)
clf = LogisticRegression()
clf.fit(vectors, newsgroups_train.target)
newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'), categories=politics)

vectors_test = vectorizer.transform(newsgroups_test.data)
pred = clf.predict(vectors_test)
metrics.f1_score(newsgroups_test.target, pred, average='binary')
test_set = pd.DataFrame(newsgroups_test.data)
test_set['class_proba'] = clf.predict_proba(vectors_test)[:,0]
test_set.columns = ['text', 'class_proba']

conn = sqlite3.connect('reuters_testX_with_proba.sqlite')
c = conn.cursor()
test_set.to_sql('reuters_test_X', conn)
conn.commit()
conn.close()

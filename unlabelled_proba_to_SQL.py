import pandas as pd
import numpy as np
import sqlite3

conn = sqlite3.connect('mini_testX_with_proba.sqlite')
c = conn.cursor()
# Create table
#c.execute('CREATE TABLE mini_testX'\
#'(review TEXT, class_proba INTEGER, date TEXT)')
df = pd.read_csv('IMDB_unlabelled_with_proba.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
df.to_sql('mini_testX', conn)
conn.commit()
conn.close()
from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np

# import HashingVectorizer from local dir
from vectorizer import vect

app = Flask(__name__)


cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir,
                 'pkl_objects',
                 'Reuters20-classifier.pkl'), 'rb'))
db = os.path.join(cur_dir, 'reviews.sqlite')
db2 = os.path.join(cur_dir, 'reuters_testX_with_proba.sqlite')


ordered_weights_dict = pickle.load(open(os.path.join(cur_dir,'pkl_objects','ordered_weights_dict.pkl'), 'rb'))

def classify(document):
    label = {0: 'negative', 1: 'positive'}
    X = vect.transform([document])
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return label[y], proba

def train(document, y):
    X = vect.transform([document])
    clf.partial_fit(X, [y])

def sqlite_entry(path, document, y):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO review_db (review, sentiment, date)"\
    " VALUES (?, ?, DATETIME('now'))", (document, y))
    conn.commit()
    conn.close()

def train_model(document, y):
    X = vect.transform([document])
    clf.partial_fit(X, [y])

def update_class_proba(path):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    cursor = c.execute('SELECT text, indexID FROM reuters_test_X')
    all_rows = cursor.fetchall()
    X = vect.transform(x[0] for x in all_rows)
    new_proba = list(float(z) for z in clf.predict_proba(X)[:, 1])
    IDs = list(int(zz) for zz in np.arange(0, 740, 1))
    new_proba_tuple = list(zip(new_proba,IDs))
    #c.execute('ALTER TABLE reuters_test_X ADD COLUMN predict_proba_tplus1 REAL')
    #c.executemany('UPDATE reuters_test_X SET predict_proba_tplus1=? WHERE indexID=?', new_proba_tuple)
    c.executemany('UPDATE reuters_test_X SET class_proba=? WHERE indexID=?', new_proba_tuple)
    new_class = list(int(xy) for xy in clf.predict(X))
    new_class_tuple = list(zip(new_class,IDs))
    c.executemany('UPDATE reuters_test_X SET prediction=? WHERE indexID=?', new_class_tuple)
    c.execute('UPDATE reuters_test_X SET predicted_labels=\'Guns\' WHERE prediction=1')
    c.execute('UPDATE reuters_test_X SET predicted_labels=\'Middle-east\' WHERE prediction=0')
    conn.commit()
    conn.close()

# This gives the index location of the weight
def look_up_weight(word):
    return list(ordered_weights_dict.keys()).index(word)

# This takes the index location of the weight and updates it
def increase_weight(index):
    if clf.coef_[0][index] !=0:
        clf.coef_[0][index] = clf.coef_[0][index] * 10
    else:
        clf.coef_[0][index] = 1

def decrease_weight(index):
    clf.coef_[0][index] = clf.coef_[0][index] / 10

def uncertainty_sample(class_proba):
    uncertainty = abs(class_proba - 0.5)
    return uncertainty


### Forms
# Need to add a constraint to only allow single words? Or if not then to split them myself and vectorize
class ReviewForm(Form):
    feature_feedback = TextAreaField('',
                                [validators.DataRequired(),
                                validators.length(min=3)])


### Flask functions

'''
@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)
'''

@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['moviereview']
        y, proba = classify(review)
        return render_template('results.html',
                                content=review,
                                prediction=y,
                                probability=round(proba*100, 2))
    return render_template('reviewform.html', form=form)


@app.route('/thanks', methods=['POST'])
def feedback():
    feedback = request.form['feedback_button']
    review = request.form['review']
    prediction = request.form['prediction']
    inv_label = {'negative': 0, 'positive': 1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = int(not(y))
    train(review, y)
    sqlite_entry(db, review, y)
    return render_template('thanks.html')

@app.route('/update', methods=['POST'])
def update_classifier_feedback():
    feedback = request.form['update_classifier']
    article = request.form['uncertain_article']
    prediction = request.form['prediction']
    inv_label = {"Middle-east": 0, "Guns": 1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = int(not(y))
    # Retrain model with uncertain article and new (or same as before) label
    train_model(article, y)
    # Update class probabilities and labels for entire test set
    update_class_proba(db2)
    return render_template('thanks.html')

@app.route('/change-weights', methods=['POST'])
def manually_change_weights():
    feedback = request.form['feature_feedback']
    increase_weight(look_up_weight(feedback))
    return render_template('thank-you.html')


# thanks.html leads here
@app.route('/article')
def display_article():
    conn = sqlite3.connect(db2)
    conn.create_function("uncertainty_query",1,uncertainty_sample)
    c = conn.cursor()
    cursor = c.execute('SELECT predicted_labels, text, MIN(uncertainty_query(class_proba)) FROM reuters_test_X')
    items = [dict(predicted_labels=row[0], text=row[1], class_proba=row[2]) for row in cursor.fetchall()]
    #items = {predicted_labels:cursor[0], text:cursor[1], class_proba:cursor[2]}
    return render_template('article.html', items=items)


# Need a second version of thanks.html to loop back here
@app.route('/article-with-reweighting')
def display_article_manual_reweighting():
    conn = sqlite3.connect(db2)
    conn.create_function("uncertainty_query", 1, uncertainty_sample)
    c = conn.cursor()
    cursor = c.execute('SELECT predicted_labels, text, MIN(uncertainty_query(class_proba)) FROM reuters_test_X')
    items = [dict(predicted_labels=row[0], text=row[1], class_proba=row[2]) for row in cursor.fetchall()]
    form = ReviewForm(request.form)
    return render_template('article-with-reweighting.html', items=items, form=form)


if __name__ == '__main__':
    app.run(debug=True)

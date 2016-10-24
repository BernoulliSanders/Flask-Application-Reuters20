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
# Load classifier
clf = pickle.load(open(os.path.join(cur_dir,
                 'pkl_objects',
                 'RCV1_log_reg_classifier.pkl'), 'rb'))

# Read in unlabelled pool of 1000 articles from RCV1
db2 = os.path.join(cur_dir, 'RCV1.sqlite')

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

def train_model(document, y):
    X = vect.transform([document])
    clf.partial_fit(X, [y])

def update_class_proba(path):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    cursor = c.execute('SELECT text, indexID FROM RCV1_test_X')
    all_rows = cursor.fetchall()
    X = vect.transform(x[0] for x in all_rows)
    new_proba = list(float(z) for z in clf.predict_proba(X)[:, 1])
    IDs = list(int(zz) for zz in np.arange(2006, 3006, 1))
    new_proba_tuple = list(zip(new_proba,IDs))
    c.executemany('UPDATE RCV1_test_X SET class_proba=? WHERE indexID=?', new_proba_tuple)
    new_class = list(int(xy) for xy in clf.predict(X))
    new_class_tuple = list(zip(new_class,IDs))
    c.executemany('UPDATE RCV1_test_X SET prediction=? WHERE indexID=?', new_class_tuple)
    c.execute('UPDATE RCV1_test_X SET predicted_labels=\'Environment and natural world\' WHERE prediction=1')
    c.execute('UPDATE RCV1_test_X SET predicted_labels=\'Defence\' WHERE prediction=0')
    conn.commit()
    conn.close()

###### End user feature feedback #######
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

###### Uncertainty sampling #####
def uncertainty_sample(class_proba):
    uncertainty = abs(class_proba - 0.5)
    return uncertainty

def uncertainty_sample_chopped(class_proba):
    uncertainty = abs(class_proba - 0.5)
    uncertainty = str(uncertainty)[:4]
    return uncertainty

### Forms
class ReviewForm(Form):
    feature_feedback = TextAreaField('',
                                [validators.DataRequired(),
                                validators.length(min=3)])

#### Flask functions ######

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

# This is called when the user clicks correct or incorrect in the active learning version
@app.route('/update', methods=['POST'])
def update_classifier_feedback():
    feedback = request.form['update_classifier']
    article = request.form['uncertain_article']
    prediction = request.form['prediction']
    inv_label = {"Defence": 0, "Environment and natural world": 1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = int(not(y))
    # Retrain model with uncertain article and new (or same as before) label
    train_model(article, y)
    # Update class probabilities and labels for entire test set
    update_class_proba(db2)
    return render_template('thanks.html')

# This is called when the user clicks correct or incorrect on the feature reweighting version
@app.route('/update-v2', methods=['POST'])
def update_classifier_feedback_v2():
    feedback = request.form['update_classifier']
    article = request.form['uncertain_article']
    prediction = request.form['prediction']
    inv_label = {"Defence": 0, "Environment and natural world": 1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = int(not(y))
    # Retrain model with uncertain article and new (or same as before) label
    train_model(article, y)
    # Update class probabilities and labels for entire test set
    update_class_proba(db2)
    return render_template('thank-you.html')

@app.route('/change-weights', methods=['POST'])
def manually_change_weights():
    feedback = request.form['feature_feedback']
    # look_up_weight finds the index location of the weight in the weight vector, increase_weight increases it
    increase_weight(look_up_weight(feedback))
    return render_template('thank-you.html')

# This function uses the uncertainty_query function defined above and uses it in a
# SQL SELECT query to display to the user the article which the model is most uncertain about
# thanks.html leads here
@app.route('/article')
def display_article():
    conn = sqlite3.connect(db2)
    conn.create_function("uncertainty_query",1,uncertainty_sample)
    c = conn.cursor()
    cursor = c.execute('SELECT predicted_labels, Headline, Text, MIN(uncertainty_query(class_proba)) FROM RCV1_test_X')
    items = [dict(predicted_labels=row[0], Headline=row[1], Text=row[2], class_proba=row[3]) for row in cursor.fetchall()]
    return render_template('article.html', items=items)

# Need a second version of thanks.html to loop back here
@app.route('/article-with-reweighting')
#@app.route('/article-with-reweighting/<article-id>')
def display_article_manual_reweighting():
    conn = sqlite3.connect(db2)
    conn.create_function("uncertainty_query", 1, uncertainty_sample)
    c = conn.cursor()
    cursor = c.execute('SELECT predicted_labels, Headline, Text, MIN(uncertainty_query(class_proba)) FROM RCV1_test_X')
    items = [dict(predicted_labels=row[0], Headline=row[1], Text=row[2], class_proba=row[3]) for row in cursor.fetchall()]
    form = ReviewForm(request.form)
    return render_template('article-with-reweighting.html', items=items, form=form)

@app.route('/menu')
def display_headlines():
    conn = sqlite3.connect(db2)
    # Could use this function for SVM I think
    # conn.create_function("ignore_sign", 1, abs)
    conn.create_function("uncertainty_query", 1, uncertainty_sample_chopped)
    c = conn.cursor()
    cursor = c.execute('SELECT Headline, uncertainty_query(class_proba), predicted_labels FROM RCV1_test_X')
    menu_items = [dict(Headline=row[0], class_proba=row[1], predicted_labels=row[2][:11]) for row in cursor.fetchall()]
    return render_template('menu.html', items=menu_items)

'''
@app.route('/menu/<article-id>')
def display_clicked_article():
    conn = sqlite3.connect(db2)
    c = conn.cursor()
    cursor = c.execute('SELECT Headline, Text FROM RCV1_test_X')
'''


if __name__ == '__main__':
    app.run(debug=True)

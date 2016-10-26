from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np
import random
import heapq

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

# This is used to name new columns in the SQLite database
count = 0

# These are used to count the number of times the study has been started and to give the new tables a unique number
active_l_participant_num = 0
feature_f_participant_num = 0

def classify(document):
    label = {0: 'negative', 1: 'positive'}
    X = vect.transform([document])
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return label[y], proba

def train_model(document, y):
    X = vect.transform([document])
    clf.partial_fit(X, [y])

def feedback_count():
    global count
    count += 1

def active_l_participant_counter():
    global active_l_participant_num
    active_l_participant_num += 1

def feature_f_participant_counter():
    global feature_f_participant_num
    feature_f_participant_num += 1

#### To be called whenever someone clicks they accept the T&C's on the welcome page, creating a new table
# Version 1 for active learning
def new_table_active(path):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    active_l_participant_counter()
    table_name = "active_learning_num"+str(active_l_participant_num)
    c.execute("CREATE TABLE "+table_name+"(ind INT, indexID INT, Headline TEXT, Text TEXT, prediction INT, predicted_labels TEXT, class_proba INT)")
    c.execute("INSERT INTO "+table_name+" SELECT * FROM RCV1_test_X;")
    conn.commit()
    conn.close()
    return display_article()

# Version 2 for feature feedback
def new_table_feature_feedback(path):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    feature_f_participant_counter()
    table_name = "feature_feedback_num"+str(feature_f_participant_num)
    c.execute("CREATE TABLE "+table_name+"(ind INT, indexID INT, Headline TEXT, Text TEXT, prediction INT, predicted_labels TEXT, class_proba INT)")
    c.execute("INSERT INTO "+table_name+" SELECT * FROM RCV1_test_X;")
    conn.commit()
    conn.close()
    # Return random article as the first one to be shown to the user. This function is only called once.
    return display_article_manual_reweighting(random.randrange(0, 1000))


'''This vectorizes all articles, calculates the updated probabilities, 
updates the class probabilities, and adds a new column for each piece 
of feedback given by the user with the class probabilities at that point in time'''
def update_class_proba_active(path,count):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    cursor = c.execute('SELECT text, indexID FROM RCV1_test_X')
    all_rows = cursor.fetchall()
    X = vect.transform(x[0] for x in all_rows)
    new_proba = list(float(z) for z in clf.predict_proba(X)[:, 1])
    IDs = list(int(zz) for zz in np.arange(0, 1000, 1))
    new_proba_tuple = list(zip(new_proba,IDs))
    new_class = list(int(xy) for xy in clf.predict(X))
    new_class_tuple = list(zip(new_class,IDs))
    # Update values in main table (which holds the articles, headlines and predicted labels)
    c.executemany('UPDATE RCV1_test_X SET class_proba=? WHERE indexID=?', new_proba_tuple)
    c.executemany('UPDATE RCV1_test_X SET prediction=? WHERE indexID=?', new_class_tuple)
    c.execute('UPDATE RCV1_test_X SET predicted_labels=\'Environment and natural world\' WHERE prediction=1')
    c.execute('UPDATE RCV1_test_X SET predicted_labels=\'Defence\' WHERE prediction=0')
    # Add new column to participant tracking table with feedback at t+x 
    table_name = "active_learning_num"+str(active_l_participant_num)
    feedback_count()
    column = "class_proba_t_plus_"+str(count)
    c.execute('ALTER TABLE '+table_name+' ADD COLUMN '+column+' REAL')
    c.executemany('UPDATE '+table_name+' SET '+column+'=? WHERE indexID=?', new_proba_tuple)
    conn.commit()
    conn.close()

def update_class_proba_feature(path,count):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    cursor = c.execute('SELECT text, indexID FROM RCV1_test_X')
    all_rows = cursor.fetchall()
    X = vect.transform(x[0] for x in all_rows)
    new_proba = list(float(z) for z in clf.predict_proba(X)[:, 1])
    IDs = list(int(zz) for zz in np.arange(0, 1000, 1))
    new_proba_tuple = list(zip(new_proba,IDs))
    new_class = list(int(xy) for xy in clf.predict(X))
    new_class_tuple = list(zip(new_class,IDs))
    # Update values in main table (which holds the articles, headlines and predicted labels)
    c.executemany('UPDATE RCV1_test_X SET class_proba=? WHERE indexID=?', new_proba_tuple)
    c.executemany('UPDATE RCV1_test_X SET prediction=? WHERE indexID=?', new_class_tuple)
    c.execute('UPDATE RCV1_test_X SET predicted_labels=\'Environment and natural world\' WHERE prediction=1')
    c.execute('UPDATE RCV1_test_X SET predicted_labels=\'Defence\' WHERE prediction=0')
    # Add new column to participant tracking table with feedback at t+x 
    table_name = "feature_feedback_num"+str(feature_f_participant_num)
    feedback_count()
    column = "class_proba_t_plus_"+str(count)
    c.execute('ALTER TABLE '+table_name+' ADD COLUMN '+column+' REAL')
    c.executemany('UPDATE '+table_name+' SET '+column+'=? WHERE indexID=?', new_proba_tuple)
    conn.commit()
    conn.close()

# This is to store the actual instance feedback given by a user
'''
def track_instance_feedback(y):
    feedback = {"incorrect": 0, "correct": 1}
    conn = sqlite3.connect(db2)
    c = conn.cursor()
    c.execute('INSERT feedback INTO ???? WHERE ?????')
'''

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
'''
def look_up_top_10_weights():
    top10 = []
    clf.coef_[0].sort(0:10)
    return heapq.nlargest(10, (clf.coef_[0]))
'''
###### Uncertainty sampling #####
def uncertainty_sample(class_proba):
    uncertainty = abs(class_proba - 0.5)
    return uncertainty

# Same as above but truncated for display on the menu
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

# This is called when the user clicks correct or incorrect in the active learning version
@app.route('/update', methods=['POST'])
def update_classifier_feedback():
    feedback = request.form['update_classifier']
    article = request.form['uncertain_article']
    prediction = request.form['prediction']
    inv_label = {"Defence": 0, "Environment and natural world": 1}
    y = inv_label[prediction]
    # track_instance_feedback(y)
    if feedback == 'Incorrect':
        y = int(not(y))
    # Retrain model with uncertain article and new (or same as before) label
    train_model(article, y)
    # Update class probabilities and labels for entire test set
    update_class_proba_active(db2,count)
    return render_template('thanks.html')

# This is called when the user clicks correct or incorrect on the feature reweighting version
@app.route('/update-v2', methods=['POST'])
def update_classifier_feedback_v2():
    feedback = request.form['update_classifier']
    article = request.form['uncertain_article']
    prediction = request.form['prediction']
    inv_label = {"Defence": 0, "Environment and natural world": 1}
    y = inv_label[prediction]
    # track_instance_feedback(y)
    if feedback == 'Incorrect':
        y = int(not(y))
    # Retrain model with uncertain article and new (or same as before) label
    train_model(article, y)
    # Update class probabilities and labels for entire test set
    update_class_proba_feature(db2,count)
    return render_template('thank-you.html')

@app.route('/change-weights', methods=['POST'])
def manually_change_weights():
    feedback = request.form['feature_feedback']
    # look_up_weight finds the index location of the weight in the weight vector, increase_weight increases it
    increase_weight(look_up_weight(feedback))
    return render_template('thank-you.html')

# This displays the active learning application. This function uses the uncertainty_query function defined above and uses it in a
# SQL SELECT query to display to the user the article which the model is most uncertain about
# thanks.html leads here
@app.route('/article')
def display_article():
    conn = sqlite3.connect(db2)
    conn.create_function("uncertainty_query",1,uncertainty_sample)
    c = conn.cursor()
    table_name = "active_learning_num"+str(active_l_participant_num)
    cursor = c.execute('SELECT predicted_labels, Headline, Text, MIN(uncertainty_query(class_proba)) FROM RCV1_test_X')
    items = [dict(predicted_labels=row[0], Headline=row[1], Text=row[2], class_proba=row[3]) for row in cursor.fetchall()]
    return render_template('article.html', items=items)

# This displays the feature feedback application
#@app.route('/article-with-reweighting')
@app.route('/article-with-reweighting/<articleid>')
def display_article_manual_reweighting(articleid):
    conn = sqlite3.connect(db2)
    conn.create_function("uncertainty_query", 1, uncertainty_sample)
    c = conn.cursor()
    # cursor = c.execute('SELECT predicted_labels, Headline, Text, MIN(uncertainty_query(class_proba)) FROM RCV1_test_X')
    # items = [dict(predicted_labels=row[0], Headline=row[1], Text=row[2], class_proba=row[3]) for row in cursor.fetchall()]
    cursor = c.execute('SELECT predicted_labels, Headline, Text, uncertainty_query(class_proba), indexID FROM RCV1_test_X WHERE indexID=?', (articleid,))
    items = [dict(predicted_labels=row[0], Headline=row[1], Text=row[2], class_proba=row[3], indexID=row[4]) for row in cursor.fetchall()]
    form = ReviewForm(request.form)
    return render_template('article-with-reweighting.html', items=items, form=form, articleid=articleid)

# Vertical menu used in an iframe on the article-with-reweighting app
@app.route('/menu')
def display_headlines():
    conn = sqlite3.connect(db2)
    # Could use this function for SVM I think
    # conn.create_function("ignore_sign", 1, abs)
    conn.create_function("uncertainty_query", 1, uncertainty_sample_chopped)
    c = conn.cursor()
    cursor = c.execute('SELECT Headline, uncertainty_query(class_proba), predicted_labels, indexID FROM RCV1_test_X')
    menu_items = [dict(Headline=row[0], class_proba=row[1], predicted_labels=row[2][:11], indexID=row[3]) for row in cursor.fetchall()]
    return render_template('menu.html', items=menu_items)

# This assigns a user to one or other study at random
@app.route('/study', methods=['POST'])
def display_app_at_random():
    page = random.randrange(0, 2)
    if page == 0:
        return new_table_active(db2)
    else:
        return new_table_feature_feedback(db2)

'''
@app.route('/menu/<article-id>')
def display_clicked_article():
    conn = sqlite3.connect(db2)
    c = conn.cursor()
    cursor = c.execute('SELECT Headline, Text FROM RCV1_test_X')
'''

@app.route('/')
def welcome_page():
    return render_template('opt-in.html')


if __name__ == '__main__':
    app.run(debug=True)

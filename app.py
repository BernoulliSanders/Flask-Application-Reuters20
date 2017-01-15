from flask import Flask, render_template, request, Markup, session
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import re
import numpy as np
import pandas as pd
import random
from operator import itemgetter
from vectorizer import vect, tokenizer
from collections import OrderedDict
import datetime
import random
import lime
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline


app = Flask(__name__)

app.secret_key = ''

cur_dir = os.path.dirname(__file__)
# Load classifier
pre_study_clf = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'RCV1_log_reg_GSCI_GHEA.pkl'), 'rb'))

# This is done to make a unique classifier for every user, but with the same starting point
#clf = pre_study_clf


pre_study_bow_vect = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'TF_IDF_vect.pkl'), 'rb'))

# Read in unlabelled pool of 1000 articles from RCV1
db2 = os.path.join(cur_dir, 'RCV1.sqlite')


# Lime
#explainer = LimeTextExplainer()
#pipeline = make_pipeline(bow_vect, clf)


# This is used to name new columns in the SQLite database
#count = 1

#feedback_given = 0

# These are used to give the new tables a unique number

#username = ""

#table_name = ""

#train_set_name = ""

'''def generate_number():
    global username
    username = np.random.randint(0, 100000000)
    return username

def feedback_count():
    global count
    count += 1
'''
def feedback_count():
  try:
    session['count'] += 1
  except KeyError:
    session['count'] = 1

'''
def feedback_given_count():
    global feedback_given
    feedback_given += 1
'''

def feedback_given_count():
  try:
    session['feedback_given'] += 1
  except KeyError:
    session['feedback_given'] = 1

# This is called every time a user gives feedback on an article. It reads the column which stores all the article and headline of all
# articles in the training set, fit transforms them, and then fits the classifier to them
def train_model_v2():
    conn = sqlite3.connect('RCV1_train.sqlite')
    #c = conn.cursor()
    train_set_name = "training_set"+str(session['name'])
    X = pd.read_sql("SELECT Full_Text FROM "+train_set_name+";", conn)
    dest = os.path.join('pkl_objects')
    if not os.path.exists(dest):
        os.makedirs(dest)
    bow_vect = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'TF_IDF_vect'+str(session['name'])+'.pkl'), 'rb'))
    X = bow_vect.fit_transform(X['Full_Text'])
    y = pd.read_sql("SELECT label FROM "+train_set_name+";",conn)
    dest = os.path.join('pkl_objects')
    clf = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'RCV1_log_reg_GSCI_GHEA'+str(session['name'])+'.pkl'), 'rb'))
    #clf = session['clf']
    clf.fit(X, y.values.ravel())
    conn.close()
    pickle.dump(bow_vect,open(os.path.join(dest, 'TF_IDF_vect'+str(session['name'])+'.pkl'), 'wb'), protocol=4)
    pickle.dump(clf,open(os.path.join(dest, 'RCV1_log_reg_GSCI_GHEA'+str(session['name'])+'.pkl'), 'wb'), protocol=4)


#This copies the original training set and adds newly labelled instances to it. One table per participant.
def create_new_training_set_table():
    conn = sqlite3.connect('RCV1_train.sqlite')
    c = conn.cursor()
    # global train_set_name
    train_set_name = "training_set"+str(session['name'])
    c.execute("CREATE TABLE "+train_set_name+"(ind INT, Headline TEXT, Text TEXT, Label INT, Full_Text TEXT)")
    c.execute("INSERT INTO "+train_set_name+" SELECT * FROM RCV1_training_set")
    conn.commit()
    conn.close()

# Add article to training set
def add_to_training_set(ind, Headline, Text, Label):
    conn = sqlite3.connect('RCV1_train.sqlite')
    train_set_name = "training_set"+str(session['name'])
    c = conn.cursor()
    Full_Text = Headline + " " + Text
    c.execute("INSERT INTO "+train_set_name+" VALUES (?, ?, ?, ?, ?)", (ind, Headline, Text, Label, Full_Text))
    conn.commit()
    conn.close()
    remove_from_test_set(ind)

# Remove newly labelled article from test set
def remove_from_test_set(ind):
    conn = sqlite3.connect('RCV1.sqlite')
    c = conn.cursor()
    table_name = "RCV1_unlabelled_pool"+str(session['name'])
    c.execute('DELETE FROM '+table_name+' WHERE indexID=?', (ind,))
    conn.commit()
    conn.close()

# This creates two new tables per feature feedback participant, one for new words added, the other for existing words which were reweighted
def create_reweighting_table():
    conn = sqlite3.connect('RCV1.sqlite')
    c = conn.cursor()
    #table_name = "weights_changed_"+str(username)
    table_name = "weights_changed_"+str(session['name'])
    c.execute("CREATE TABLE "+table_name+"(feature_label TEXT, word TEXT, LIME_or_coef Text, feedback_iteration INT, articleid INT)")
    #new_words_table = "new_words_added_"+str(username)
    new_words_table = "new_words_added_"+str(session['name'])
    c.execute("CREATE TABLE "+new_words_table+"(word TEXT, predicted_label TEXT, instance_label_given TEXT, feedback_iteration INT)")
    conn.commit()
    conn.close()

def insert_into_reweighting_table(feedback):
    conn = sqlite3.connect('RCV1.sqlite')
    c = conn.cursor()
    table_name = "weights_changed_"+str(session['name'])
    words = [tuple(x.split(' ')) for x in feedback]
    c.executemany("INSERT INTO "+table_name+" (feature_label, word, LIME_or_coef, feedback_iteration, articleid) VALUES (?,?,?,?,?)", words)
    conn.commit()
    conn.close()

def insert_into_new_words_added_table(word,label, instance_label_given, feedback_count):
    conn = sqlite3.connect('RCV1.sqlite')
    c = conn.cursor()
    table_name = "new_words_added_"+str(session['name'])
    c.execute("INSERT INTO "+table_name+" (word, predicted_label, instance_label_given, feedback_iteration) VALUES (?,?,?,?)", (word, label, instance_label_given, feedback_count))
    conn.commit()
    conn.close()



'''This vectorizes all articles, calculates the updated probabilities,
updates the class probabilities, and adds a new column for each piece
of feedback given by the user with the class probabilities at that point in time'''
def update_class_proba_active(path):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    cursor = c.execute('SELECT Full_Text, indexID FROM RCV1_test_X')
    #table_name = "active_learning_"+str(username)
    #cursor = c.execute('SELECT Full_Text, indexID FROM '+table_name+';')
    table_name = "RCV1_unlabelled_pool"+str(session['name'])
    all_rows = cursor.fetchall()
    bow_vect = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'TF_IDF_vect'+str(session['name'])+'.pkl'), 'rb'))
    X = bow_vect.transform(x[0] for x in all_rows)
    # Calculate new class probabilities for class 1
    # clf = session['clf']
    clf = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'RCV1_log_reg_GSCI_GHEA'+str(session['name'])+'.pkl'), 'rb'))
    new_proba = list(float(z) for z in clf.predict_proba(X)[:, 1])
    IDs = list(int(zz) for zz in np.arange(0, 1000, 1))
    new_proba_tuple = list(zip(new_proba, IDs))
    # Update predicted classes for unlabelled pool
    new_class = list(int(xy) for xy in clf.predict(X))
    new_class_tuple = list(zip(new_class,IDs))
    # Add new column to participant tracking table with feedback at t+x
    c.executemany('UPDATE '+table_name+' SET class_proba=? WHERE indexID=?', new_proba_tuple)
    c.executemany('UPDATE '+table_name+' SET prediction=? WHERE indexID=?', new_class_tuple)
    c.execute('UPDATE '+table_name+' SET predicted_labels=\'Health\' WHERE prediction=1')
    c.execute('UPDATE '+table_name+' SET predicted_labels=\'Science\' WHERE prediction=0')
    feedback_count()
    column = "class_proba_t_plus_"+str(session['count'])
    c.execute('ALTER TABLE '+table_name+' ADD COLUMN '+column+' REAL')
    c.executemany('UPDATE '+table_name+' SET '+column+'=? WHERE indexID=?', new_proba_tuple)
    conn.commit()
    conn.close()

def update_class_proba_feature(path):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    cursor = c.execute('SELECT Full_Text, indexID FROM RCV1_test_X')
    #table_name = "feature_feedback_"+str(username)
    #cursor = c.execute('SELECT Full_Text, indexID FROM '+table_name+';')
    all_rows = cursor.fetchall()
    bow_vect = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'TF_IDF_vect'+str(session['name'])+'.pkl'), 'rb'))
    X = bow_vect.transform(x[0] for x in all_rows)
    clf = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'RCV1_log_reg_GSCI_GHEA'+str(session['name'])+'.pkl'), 'rb'))
    new_proba = list(round(float(z),3) for z in clf.predict_proba(X)[:, 1])
    IDs = list(int(zz) for zz in np.arange(0, 1000, 1))
    new_proba_tuple = list(zip(new_proba,IDs))
    new_class = list(int(xy) for xy in clf.predict(X))
    new_class_tuple = list(zip(new_class,IDs))
    # Add new column to participant tracking table with feedback at t+x
    table_name = "RCV1_unlabelled_pool"+str(session['name'])
    c.executemany('UPDATE '+table_name+' SET class_proba=? WHERE indexID=?', new_proba_tuple)
    c.executemany('UPDATE '+table_name+' SET prediction=? WHERE indexID=?', new_class_tuple)
    c.execute('UPDATE '+table_name+' SET predicted_labels=\'Health\' WHERE prediction=1')
    c.execute('UPDATE '+table_name+' SET predicted_labels=\'Science\' WHERE prediction=0')
    feedback_count()
    column = "class_proba_t_plus_"+str(session['count'])
    c.execute('ALTER TABLE '+table_name+' ADD COLUMN '+column+' REAL')
    c.executemany('UPDATE '+table_name+' SET '+column+'=? WHERE indexID=?', new_proba_tuple)
    conn.commit()
    conn.close()


# Returns a list of tuples of the top 10 weights in the string, used to find top ten weights for the article currently being displayed to the end user
'''
def look_up_word(article, pred_class):
    matching_iloc = []
    matching_word = []
    for i in article:
        if bow_vect.vocabulary_.get(i) != None:
            matching_iloc.append(bow_vect.vocabulary_.get(i))
            matching_word.append(i)
    matching_weights = []
    for word in matching_iloc:
            matching_weights.append(clf.coef_[0][word])
    weights_dict = dict(zip(matching_word,matching_weights))
    if pred_class == "[('Science',)]":
        weights_dict = dict(sorted(weights_dict.items(), key=itemgetter(1), reverse = False)[0:10])
    else:
        weights_dict = dict(sorted(weights_dict.items(), key=itemgetter(1), reverse = True)[0:10])
    return weights_dict
'''


###### Active learning #####
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
                                validators.length(min=1)])
    submit_percentage_feature_feedback = TextAreaField('',
                                [validators.DataRequired(),
                                validators.length(min=1)])
    username = TextAreaField('',
                                [validators.DataRequired(),
                                validators.length(min=1)])
    new_instance = TextAreaField('',
                                [validators.DataRequired(),
                                validators.length(min=1)])


##### ##### ##### ##### #####
#### Flask functions ######

# First function called when a user views the base URL
@app.route('/')
def welcome_page():
    session['count'] = 0
    session['feedback_given'] = 0
    session['name'] = np.random.randint(0, 100000000)
    return render_template('opt-in.html')

# This assigns a user to one or other study at random
@app.route('/study', methods=['POST'])
def display_app_at_random():
    checkbox = request.form['age']
    if checkbox == "confirmed":
        session['page'] = int(datetime.datetime.now().strftime('%S')) % 2
    if session['page'] == 0:
        return render_template('feature-labelling-user-instructions.html')
    elif session['page'] == 1: 
        return render_template('active-learning-user-instructions.html')
    else:
        return render_template('opt-in.html')


@app.route('/finish', methods=['POST'])
def display_end_page():
    return render_template('thank-you-for-your-participation.html')

#### To be called whenever someone clicks they accept the T&C's on the welcome page, creating a new table
# Version 1 for active learning
@app.route('/active-learning', methods=['POST'])
def new_table_active():
    # Copy pre-existing clf to a new variable. I just need to create the file and then reference the name correctly whenever used.
    #global clf
    #clf = pre_study_clf
    #session['clf'] = pre_study_clf
    dest = os.path.join('pkl_objects')
    if not os.path.exists(dest):
        os.makedirs(dest)
    pickle.dump(pre_study_bow_vect,open(os.path.join(dest, 'TF_IDF_vect'+str(session['name'])+'.pkl'), 'wb'), protocol=4)
    pickle.dump(pre_study_clf,open(os.path.join(dest, 'RCV1_log_reg_GSCI_GHEA'+str(session['name'])+'.pkl'), 'wb'), protocol=4)
    confirmed = request.form['proceed_with_study']
    conn = sqlite3.connect(db2)
    c = conn.cursor()
    # global table_name
    table_name = "RCV1_unlabelled_pool"+str(session['name'])
    #table_name = "active_learning_"+str(session['name'])
    c.execute("CREATE TABLE "+table_name+"(ind INT, indexID INT, Label INT, Headline TEXT, Text TEXT, Full_Text TEXT, prediction INT, predicted_labels TEXT, class_proba INT)")
    c.execute("INSERT INTO "+table_name+" SELECT * FROM RCV1_test_X;")
    conn.commit()
    conn.close()
    create_new_training_set_table()
    return display_article()

# Version 2 for feature feedback
@app.route('/feature-feedback', methods=['POST'])
def new_table_feature_feedback():
    #global clf
    dest = os.path.join('pkl_objects')
    if not os.path.exists(dest):
        os.makedirs(dest)
    pickle.dump(pre_study_bow_vect,open(os.path.join(dest, 'TF_IDF_vect'+str(session['name'])+'.pkl'), 'wb'), protocol=4)
    pickle.dump(pre_study_clf,open(os.path.join(dest, 'RCV1_log_reg_GSCI_GHEA'+str(session['name'])+'.pkl'), 'wb'), protocol=4)
    conn = sqlite3.connect(db2)
    c = conn.cursor()
    create_reweighting_table()
    # global table_name
    # table_name = "feature_feedback_"+str(session['name'])
    table_name = "RCV1_unlabelled_pool"+str(session['name'])
    c.execute("CREATE TABLE "+table_name+"(ind INT, indexID INT, Label INT, Headline TEXT, Text TEXT, Full_Text TEXT, prediction INT, predicted_labels TEXT, class_proba INT)")
    c.execute("INSERT INTO "+table_name+" SELECT * FROM RCV1_test_X;")
    conn.commit()
    conn.close()
    create_new_training_set_table()
    # Return random article as the first one to be shown to the user. This is only called once.
    return display_article_manual_reweighting(random.randrange(0, 500))

# This displays the instance based active learning version. This function uses the uncertainty_query function defined above and uses it in a
# SQL SELECT query to display to the user the article which the model is most uncertain about
@app.route('/article')
def display_article():
    conn = sqlite3.connect(db2)
    conn.create_function("uncertainty_query",1,uncertainty_sample)
    c = conn.cursor()
    # table_name = "active_learning_"+str(session['name'])
    table_name = "RCV1_unlabelled_pool"+str(session['name'])
    cursor = c.execute('SELECT predicted_labels, Headline, Text, MIN(uncertainty_query(class_proba)), indexID FROM '+table_name+';')
    items = [dict(predicted_labels=row[0], Headline=row[1], Text=row[2], class_proba=row[3], indexID=row[4]) for row in cursor.fetchall()]
    return render_template('article.html', items=items, feedback_given=session['feedback_given'])

# This displays the feature feedback application
@app.route('/article-with-reweighting/<articleid>')
def display_article_manual_reweighting(articleid):
    conn = sqlite3.connect(db2)
    conn.create_function("uncertainty_query", 1, uncertainty_sample)
    c = conn.cursor()
    table_name = "RCV1_unlabelled_pool"+str(session['name'])
    cursor2 = c.execute('SELECT predicted_labels FROM '+table_name+' WHERE INDEXID=?', (articleid,))
    pred_class = cursor2.fetchall()
    cursor = c.execute('SELECT predicted_labels, Headline, Text, uncertainty_query(class_proba), indexID FROM '+table_name+' WHERE indexID=?', (articleid,))
    items = [dict(predicted_labels=row[0], Headline=row[1], Text=row[2], class_proba=row[3], indexID=row[4]) for row in cursor.fetchall()]
    form = ReviewForm(request.form)
    bow_vect = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'TF_IDF_vect'+str(session['name'])+'.pkl'), 'rb'))
    clf = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'RCV1_log_reg_GSCI_GHEA'+str(session['name'])+'.pkl'), 'rb'))
    weight_coef_dict = dict(zip(bow_vect.get_feature_names(), clf.coef_[0]))
    cursor3 = c.execute('SELECT Text FROM '+table_name+' WHERE INDEXID=?', (articleid,))
    article = cursor3.fetchall()
    article = str(article)
    #article = re.sub("\\ \'","'", article)
    # Lime
    explainer = LimeTextExplainer()
    pipeline = make_pipeline(bow_vect, clf)
    exp = explainer.explain_instance(article, pipeline.predict_proba, num_features=10)
    lime_features = dict(exp.as_list())
    pred_class = str(pred_class)
    # top_ten_from_article = look_up_word(tokenizer(article), pred_class)
    for w in lime_features.keys():
        article = article.replace(' '+w+' ', ' <mark>'+w+'</mark> ')
        #article = re.sub(' '+w+' ', ' <mark>'+w+'</mark> ', article, flags=re.IGNORECASE)
    article = article[3:-4] # This removes the quote marks
    article = Markup(article)
    if pred_class == "[('Science',)]":
        topten = OrderedDict(sorted(weight_coef_dict.items(), key=itemgetter(1), reverse = False)[0:10])
        topten_for_chart = OrderedDict(reversed(list(topten.items())))
        lime_features = OrderedDict(sorted(lime_features.items(), key=itemgetter(1), reverse = False))
        lime_features_for_chart = OrderedDict(reversed(list(lime_features.items())))
        return render_template('article-with-reweighting.html', items=items, form=form, articleid=articleid, pred_class=pred_class, top_ten_from_article=topten, topten_for_chart_article=topten_for_chart, article=article, feedback_given=session['feedback_given'], lime_features=lime_features, lime_features_for_chart=lime_features_for_chart)
    else:
        topten = OrderedDict(sorted(weight_coef_dict.items(), key=itemgetter(1), reverse = True)[0:10])
        topten_for_chart = OrderedDict(reversed(list(topten.items())))
        lime_features = OrderedDict(sorted(lime_features.items(), key=itemgetter(1), reverse = True))
        lime_features_for_chart = OrderedDict(reversed(list(lime_features.items())))
        return render_template('article-with-reweighting.html', items=items, form=form, articleid=articleid, pred_class=pred_class,top_ten_from_article=topten, topten_for_chart_article=topten_for_chart, article=article, feedback_given=session['feedback_given'], lime_features=lime_features, lime_features_for_chart=lime_features_for_chart)


# Vertical menu used in an iframe on the article-with-reweighting app
@app.route('/menu')
def display_headlines():
    conn = sqlite3.connect(db2)
    conn.create_function("uncertainty_query", 1, uncertainty_sample_chopped)
    c = conn.cursor()
    table_name = "RCV1_unlabelled_pool"+str(session['name'])
    cursor = c.execute('SELECT Headline, uncertainty_query(class_proba), predicted_labels, indexID FROM '+table_name+' ORDER BY uncertainty_query(class_proba) ASC;')
    menu_items = [dict(Headline=row[0], class_proba=row[1], predicted_labels=row[2][:11], indexID=row[3]) for row in cursor.fetchall()]
    return render_template('menu.html', items=menu_items)


# This is called when the user clicks correct or incorrect in the active learning version
@app.route('/update', methods=['POST'])
def update_classifier_feedback():
    feedback = request.form['update_classifier']
    article = request.form['uncertain_article']
    prediction = request.form['prediction']
    headline = request.form['article_headline']
    articleid = request.form['articleid']
    inv_label = {"Science": 0, "Health": 1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = int(not(y))
    add_to_training_set(articleid,headline,article,y)
    train_model_v2()
    feedback_given_count()
    # Update class probabilities and labels for entire test set
    update_class_proba_active(db2)
    return render_template('thanks.html')

# This is called when the user clicks correct or incorrect on the feature reweighting version
@app.route('/update-v2', methods=['POST'])
def update_classifier_feedback_v2():
    feedback = request.form['update_classifier']
    article = request.form['uncertain_article']
    headline = request.form['article_headline']
    prediction = request.form['prediction']
    articleid = request.form['articleid']
    new_words = request.form['new_instance']
    inv_label = {"Science": 0, "Health": 1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = int(not(y))
    # If the prediction was correct, add the new words to the instance and the TF-IDF matrix
    if feedback == 'Correct':
        article = article + " " + new_words
    # Add newly labelled article to training set
    add_to_training_set(articleid,headline,article,y)
    # Retrain model with uncertain article and new (or same as before) label
    train_model_v2()
    # Update class probabilities and labels for entire test set
    update_class_proba_feature(db2)
    feedback_given_count()
    # session['feedback_given'] += 1
    feedback_list = []
    for i in range(1,11):
        feedback = request.form['change_weight_'+str(i)]
        #if "ignore" not in feedback:
        feedback_list.append(feedback + " " + str(session['feedback_given']) + " " + articleid)
    for i in range(1,11):
        feedback = request.form['change_weight_overall_'+str(i)]
        #if "ignore" not in feedback:
        feedback_list.append(feedback + " " + str(session['feedback_given'])+  " " + articleid)       
    insert_into_reweighting_table(feedback_list)
    if len(new_words) > 0:
        insert_into_new_words_added_table(new_words, prediction, y, session['feedback_given'])
        return render_template('thank-you.html')
    else:
        return render_template('thank-you.html')


if __name__ == '__main__':
    app.run(debug=True)




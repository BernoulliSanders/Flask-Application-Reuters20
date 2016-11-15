from flask import Flask, render_template, request, Markup, session
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import re
import numpy as np
import pandas as pd
import random
import heapq
from operator import itemgetter
from vectorizer import vect, tokenizer
from collections import OrderedDict
from datetime import timedelta
import random

app = Flask(__name__)

cur_dir = os.path.dirname(__file__)
# Load classifier 
pre_study_clf = pickle.load(open(os.path.join(cur_dir,
                 'pkl_objects',
                 'RCV1_log_reg_GDEF_GDIS.pkl'), 'rb'))

clf = pre_study_clf


bow_vect = pickle.load(open(os.path.join(cur_dir,
                 'pkl_objects',
                 'TF_IDF_vect.pkl'), 'rb'))

# Read in unlabelled pool of 1000 articles from RCV1
db2 = os.path.join(cur_dir, 'RCV1.sqlite')

#ordered_weights_dict = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'ordered_weights_dict.pkl'), 'rb'))

# This is used to name new columns in the SQLite database
count = 1

feedback_given = 0

# These are used to count the number of times the study has been started and to give the new tables a unique number
active_l_participant_num = 0
feature_f_participant_num = 0

username = np.random.randint(0, 100000000)

table_name = ""

train_set_name = ""


def train_model_v2():
    conn = sqlite3.connect('RCV1_train.sqlite')
    c = conn.cursor()
    # Need to update this so that it's taken from the table variable name
    X = pd.read_sql("SELECT Full_Text FROM "+train_set_name+";", conn)
    X = bow_vect.fit_transform(X['Full_Text'])
    y = pd.read_sql("SELECT label FROM "+train_set_name+";",conn)
    clf.fit(X, y.values.ravel())
    conn.close()
    dest = os.path.join('pkl_objects')
    if not os.path.exists(dest):
        os.makedirs(dest)
    pickle.dump(bow_vect,open(os.path.join(dest, 'TF_IDF_vect'+str(username)+'.pkl'), 'wb'), protocol=4)
    pickle.dump(clf,open(os.path.join(dest, 'RCV1_log_reg_GDEF_GDIS'+str(username)+'.pkl'), 'wb'), protocol=4)


def feedback_count():
    global count
    count += 1

def feedback_given_count():
    global feedback_given
    feedback_given += 1

'''
def active_l_participant_counter():
    global active_l_participant_num
    active_l_participant_num += 1

def feature_f_participant_counter():
    global feature_f_participant_num
    feature_f_participant_num += 1
'''




#This copies the original training set and adds newly labelled instances to it. One table per participant.
def create_new_training_set_table():
    conn = sqlite3.connect('RCV1_train.sqlite')
    c = conn.cursor()
    global train_set_name
    train_set_name = table_name+"training_set"
    c.execute("CREATE TABLE "+train_set_name+"(ind INT, Headline TEXT, Text TEXT, Label INT, Full_Text TEXT)")
    c.execute("INSERT INTO "+train_set_name+" SELECT * FROM RCV1_training_set")
    conn.commit()
    conn.close()

# Add article to training set
def add_to_training_set(index, Headline, Text, Label):
    conn = sqlite3.connect('RCV1_train.sqlite')
    c = conn.cursor()
    Full_Text = Headline + " " + Text
    c.execute("INSERT INTO "+train_set_name+" VALUES (?, ?, ?, ?, ?)", (index, Headline, Text, Label, Full_Text))
    conn.commit()
    conn.close()
    remove_from_test_set(index)

# Remove newly labelled article from training set
def remove_from_test_set(index):
    conn = sqlite3.connect('RCV1.sqlite')
    c = conn.cursor()
    c.execute('DELETE FROM '+table_name+' WHERE indexID=?', (index,))
    conn.commit()
    conn.close()

# This tracks the feature feedback given by the participant
def create_reweighting_table(username):
    conn = sqlite3.connect('RCV1.sqlite')
    c = conn.cursor()
    table_name = "weights_changed_"+str(username)
    c.execute("CREATE TABLE "+table_name+"(word TEXT, change TEXT, feedback_iteration INT)")
    new_words_table = "new_words_added_"+str(username)
    c.execute("CREATE TABLE "+new_words_table+"(word TEXT, label TEXT, feedback_iteration INT)")
    conn.commit()
    conn.close()


def insert_into_reweighting_table(feedback):
    conn = sqlite3.connect('RCV1.sqlite')
    c = conn.cursor()
    table_name = "weights_changed_"+str(username)
    words = [tuple(x.split(' ')) for x in feedback]
    c.executemany("INSERT INTO "+table_name+" (word, change, feedback_iteration) VALUES (?,?,?)", words)
    conn.commit()
    conn.close()

def insert_into_new_words_added_table(word,label,feedback_count):
    conn = sqlite3.connect('RCV1.sqlite')
    c = conn.cursor()
    table_name = "new_words_added_"+str(username)
    c.execute("INSERT INTO "+table_name+" (word, label, feedback_iteration) VALUES (?,?,?)", (word, label, feedback_count))
    conn.commit()
    conn.close()


'''
def insert_new_instance(article_id, new_instance, label):
    conn = sqlite3.connect('RCV1_train.sqlite')
    c = conn.cursor()
    c.execute("INSERT INTO "+train_set_name+" VALUES (?,?,?,?,?)", (article_id, 0, 0, label, new_instance))
    conn.commit()
    conn.close()
'''


'''This vectorizes all articles, calculates the updated probabilities, 
updates the class probabilities, and adds a new column for each piece 
of feedback given by the user with the class probabilities at that point in time'''
def update_class_proba_active(path,count):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    cursor = c.execute('SELECT text, indexID FROM RCV1_test_X')
    all_rows = cursor.fetchall()
    X = bow_vect.transform(x[0] for x in all_rows)
    # Calculate new class probabilities
    new_proba = list(float(z) for z in clf.predict_proba(X)[:, 1])
    IDs = list(int(zz) for zz in np.arange(0, 1000, 1))
    new_proba_tuple = list(zip(new_proba, IDs))
    # Update predicted classes for unlabelled pool
    new_class = list(int(xy) for xy in clf.predict(X))
    new_class_tuple = list(zip(new_class,IDs))
    # Update values in main table (which holds the articles, headlines and predicted labels) (N.B. not updating these values in user feedback table)
    c.executemany('UPDATE RCV1_test_X SET class_proba=? WHERE indexID=?', new_proba_tuple)
    c.executemany('UPDATE RCV1_test_X SET prediction=? WHERE indexID=?', new_class_tuple)
    c.execute('UPDATE RCV1_test_X SET predicted_labels=\'Environment and natural world\' WHERE prediction=1')
    c.execute('UPDATE RCV1_test_X SET predicted_labels=\'Defence\' WHERE prediction=0')
    # Add new column to participant tracking table with feedback at t+x 
    table_name = "active_learning_"+str(username)
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
    X = bow_vect.transform(x[0] for x in all_rows)
    new_proba = list(round(float(z),3) for z in clf.predict_proba(X)[:, 1])
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
    table_name = "feature_feedback_"+str(username)
    feedback_count()
    column = "class_proba_t_plus_"+str(count)
    c.execute('ALTER TABLE '+table_name+' ADD COLUMN '+column+' REAL')
    c.executemany('UPDATE '+table_name+' SET '+column+'=? WHERE indexID=?', new_proba_tuple)
    conn.commit()
    conn.close()


###### End user feature feedback #######

##### Functions for free text form ######
# This gives the index location of the weight
'''
def look_up_weight(word):
    return list(ordered_weights_dict.keys()).index(word)

# This takes the index location of the weight and updates it.
# So far this only updates the weight in the classifier coefficient. I will need to update it in the ordered weights dict too.

def increase_weight(index):
    if clf.coef_[0][index] !=0:
        clf.coef_[0][index] = clf.coef_[0][index] * 10
    else:
        clf.coef_[0][index] = 1

def decrease_weight(index):
    clf.coef_[0][index] = clf.coef_[0][index] / 10
'''

def change_weight(word, percentage):
    percentage = int(percentage)
    index = bow_vect.vocabulary_.get(word)
    if index != None:
        clf.coef_[0][index] = clf.coef_[0][index] + clf.coef_[0][index] * (percentage/100)
 

# This updates the ordered weights dict
def increase_weight_in_weights_dict_perc(word, percentage):
    percentage = int(percentage)
    if ordered_weights_dict[word] !=0:
        ordered_weights_dict[word] = ordered_weights_dict[word] + ordered_weights_dict[word] * (percentage/100)
    elif percentage > 0:
        ordered_weights_dict[word] = 10
    else:
        ordered_weights_dict[word] = -10


# Returns a list of tuples of the top 10 weights in the string, used to find top ten weights for the article currently being displayed to the end user   
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
    if pred_class == "[('Defence',)]":
        weights_dict = dict(sorted(weights_dict.items(), key=itemgetter(1), reverse = False)[0:10])
    else:
        weights_dict = dict(sorted(weights_dict.items(), key=itemgetter(1), reverse = True)[0:10])
    return weights_dict 



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
    global count
    count = 1
    global feedback_given
    feedback_given = 0
    return render_template('opt-in.html')

# This assigns a user to one or other study at random
@app.route('/study', methods=['POST'])
def display_app_at_random():
    checkbox = request.form['age']
    if checkbox == "confirmed":
        page = random.randrange(0, 2)
    if page == 0:
        return render_template('active-learning-user-instructions.html')
        #return new_table_active(db2, username)
    elif page == 1:
        return render_template('feature-labelling-user-instructions.html')
        #return new_table_feature_feedback(db2, username)
    else:
        return render_template('opt-in.html')


#### To be called whenever someone clicks they accept the T&C's on the welcome page, creating a new table
# Version 1 for active learning
@app.route('/active-learning', methods=['POST'])
def new_table_active():
    confirmed = request.form['proceed_with_study']
    conn = sqlite3.connect(db2)
    c = conn.cursor()
    # active_l_participant_counter()
    global table_name
    table_name = "active_learning_"+str(username)
    c.execute("CREATE TABLE "+table_name+"(ind INT, indexID INT, Headline TEXT, Text TEXT, Full_Text TEXT, prediction INT, predicted_labels TEXT, class_proba INT)")
    c.execute("INSERT INTO "+table_name+" SELECT * FROM RCV1_test_X;")
    conn.commit()
    conn.close()
    create_new_training_set_table()
    return display_article()

# Version 2 for feature feedback
@app.route('/feature-feedback', methods=['POST'])
def new_table_feature_feedback():
    conn = sqlite3.connect(db2)
    c = conn.cursor()
    # feature_f_participant_counter()
    create_reweighting_table(username)
    global table_name
    table_name = "feature_feedback_"+str(username)
    c.execute("CREATE TABLE "+table_name+"(ind INT, indexID INT, Headline TEXT, Text TEXT, Full_Text TEXT, prediction INT, predicted_labels TEXT, class_proba INT)")
    c.execute("INSERT INTO "+table_name+" SELECT * FROM RCV1_test_X;")
    conn.commit()
    conn.close()
    create_new_training_set_table()
    # Return random article as the first one to be shown to the user. This function is only called once.
    return display_article_manual_reweighting(random.randrange(0, 500))

# This displays the active learning version. This function uses the uncertainty_query function defined above and uses it in a
# SQL SELECT query to display to the user the article which the model is most uncertain about
@app.route('/article')
def display_article():
    conn = sqlite3.connect(db2)
    conn.create_function("uncertainty_query",1,uncertainty_sample)
    c = conn.cursor()
    #table_name = "active_learning_num"+str(active_l_participant_num)
    cursor = c.execute('SELECT predicted_labels, Headline, Text, MIN(uncertainty_query(class_proba)), indexID FROM '+table_name+';')
    items = [dict(predicted_labels=row[0], Headline=row[1], Text=row[2], class_proba=row[3], indexID=row[4]) for row in cursor.fetchall()]
    return render_template('article.html', items=items, feedback_given=feedback_given)

# This displays the feature feedback application
#@app.route('/article-with-reweighting')
@app.route('/article-with-reweighting/<articleid>')
def display_article_manual_reweighting(articleid):
    conn = sqlite3.connect(db2)
    conn.create_function("uncertainty_query", 1, uncertainty_sample)
    c = conn.cursor()
    cursor2 = c.execute('SELECT predicted_labels FROM RCV1_test_X WHERE INDEXID=?', (articleid,))    
    pred_class = cursor2.fetchall()
    cursor = c.execute('SELECT predicted_labels, Headline, Text, uncertainty_query(class_proba), indexID FROM RCV1_test_X WHERE indexID=?', (articleid,))
    items = [dict(predicted_labels=row[0], Headline=row[1], Text=row[2], class_proba=row[3], indexID=row[4]) for row in cursor.fetchall()]
    form = ReviewForm(request.form)
    weight_coef_dict = dict(zip(bow_vect.get_feature_names(), clf.coef_[0]))
    cursor3 = c.execute('SELECT Text FROM RCV1_test_X WHERE INDEXID=?', (articleid,))    
    article = cursor3.fetchall()
    article = str(article)
    pred_class = str(pred_class)
    top_ten_from_article = look_up_word(tokenizer(article), pred_class)
    for w in top_ten_from_article.keys():
        #article = article.replace(' '+w+' ', ' <mark>'+w+'</mark> ')
        article = re.sub(' '+w+' ', ' <mark>'+w+'</mark> ', article, flags=re.IGNORECASE)
    article = article[3:-4] # This removes the quote marks
    article = Markup(article)
    if pred_class == "[('Defence',)]":
        topten = OrderedDict(sorted(weight_coef_dict.items(), key=itemgetter(1), reverse = False)[0:10])
        topten_for_chart = OrderedDict(reversed(list(topten.items())))
        top_ten_from_article = OrderedDict(sorted(top_ten_from_article.items(), key=itemgetter(1), reverse = False)[0:10])
        topten_for_chart_article = OrderedDict(reversed(list(top_ten_from_article.items())))
        return render_template('article-with-reweighting.html', items=items, form=form, articleid=articleid, topten=topten, pred_class=pred_class, top_ten_from_article=top_ten_from_article, topten_for_chart_article=topten_for_chart_article, article=article, feedback_given=feedback_given, topten_for_chart=topten_for_chart)  
    else:
        topten = OrderedDict(sorted(weight_coef_dict.items(), key=itemgetter(1), reverse = True)[0:10])
        topten_for_chart = OrderedDict(reversed(list(topten.items())))
        top_ten_from_article = OrderedDict(sorted(top_ten_from_article.items(), key=itemgetter(1), reverse = True)[0:10])
        topten_for_chart_article = OrderedDict(reversed(list(top_ten_from_article.items())))
        return render_template('article-with-reweighting.html', items=items, form=form, articleid=articleid, topten=topten, pred_class=pred_class,top_ten_from_article=top_ten_from_article, topten_for_chart_article=topten_for_chart_article, article=article, feedback_given=feedback_given, topten_for_chart=topten_for_chart)


# Vertical menu used in an iframe on the article-with-reweighting app
@app.route('/menu')
def display_headlines():
    conn = sqlite3.connect(db2)
    # Could use this function for SVM I think
    # conn.create_function("ignore_sign", 1, abs)
    conn.create_function("uncertainty_query", 1, uncertainty_sample_chopped)
    c = conn.cursor()
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
    inv_label = {"Defence": 0, "Environment and natural world": 1}
    y = inv_label[prediction]
    # track_instance_feedback(y)
    if feedback == 'Incorrect':
        y = int(not(y))
    add_to_training_set(articleid,headline,article,y)
    # Retrain model with uncertain article and new (or same as before) label - should really update this to incorporate the headline too.
    train_model_v2()
    # Update class probabilities and labels for entire test set
    update_class_proba_active(db2,count)
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
    inv_label = {"Defence": 0, "Environment and natural world": 1}
    y = inv_label[prediction]
    # track_instance_feedback(y)
    if feedback == 'Incorrect':
        y = int(not(y))
    article = article + " " + new_words
    # Add newly labelled article to training set
    add_to_training_set(articleid,headline,article,y)
    # Retrain model with uncertain article and new (or same as before) label
    train_model_v2()
    # Update class probabilities and labels for entire test set
    update_class_proba_feature(db2,count)
    feedback_given_count()
    feedback_list = []
    for i in range(1,11):
        feedback = request.form['change_weight_'+str(i)]
        if "ignore" not in feedback:
            feedback_list.append(feedback + " " + str(count))
    for i in range(1,11):
        feedback = request.form['change_weight_overall_'+str(i)]
        if "ignore" not in feedback:
            feedback_list.append(feedback + " " + str(count))
    insert_into_reweighting_table(feedback_list)
    if len(new_words) > 0:
        insert_into_new_words_added_table(new_words,y,count)
        return render_template('thank-you.html')
    else:
        return render_template('thank-you.html')

'''
@app.route('/change-weights', methods=['POST'])
def show_weights():
    feedback_list = []
    feedback_1 = request.form['change_weight_1']
    feedback_list.append(feedback_1)
    feedback_2 = request.form['change_weight_2']
    feedback_list.append(feedback_2)
    #feedback = request.form['weight_updates']
    feedback_list = str(feedback_list)
    return feedback_list
'''

# This retrieves the user feedback on features
'''
@app.route('/change-weights', methods=['POST'])
def show_weights():
    feedback_list = []
    for i in range(1,11):
        feedback = request.form['change_weight_'+str(i)]
        if "ignore" not in feedback:
            feedback_list.append(feedback + " " + str(count))
    for i in range(1,11):
        feedback = request.form['change_weight_overall_'+str(i)]
        if "ignore" not in feedback:
            feedback_list.append(feedback + " " + str(count))
    #feedback_2 = request.form['change_weight_2']
    #feedback_list.append(feedback_2)
    #feedback = request.form['weight_updates']
    insert_into_reweighting_table(feedback_list)
    #feedback_list = str(feedback_list)
    #return feedback_list
'''

'''
@app.route('/change-weights', methods=['POST'])
def manually_change_weights():
    # This pulls the contents from the feature_feedback form in article-with-reweighting
    feedback = request.form['feature_feedback']
    # look_up_weight finds the index location of the weight in the weight vector, increase_weight increases it
    increase_weight_in_weight_dict(feedback)
    increase_weight(look_up_weight(feedback)) 
    return render_template('thank-you.html')
'''
'''
# This adds the new words as a labelled instance
@app.route('/add_new_instance', methods=['POST'])
def retrieve_new_instance():
    feedback = request.form['new_instance']
    articleid = request.form['articleid']
    prediction = request.form['prediction']
    inv_label = {"Defence": 0, "Environment and natural world": 1}
    y = inv_label[prediction]
    submit = request.form['submit_new_instance']
    insert_new_instance(articleid,feedback,y)
    return display_article_manual_reweighting(articleid)
'''


if __name__ == '__main__':
    app.run(debug=True)

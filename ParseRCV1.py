#### Script for parsing desired categories from the Reuters RCV1 dataset from separate XML files to one csv, suitable for a machine learning task using Scikit-learn or similar. First column is the headline, second is the body text, third is the label
import pandas as pd
import sqlite3
import xml.etree.ElementTree as ET
import os
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn import linear_model as lm, metrics, cross_validation as cv,\
                    grid_search
            
import pickle

# Makes a list with each line separately
# Each article (returned as row) is stored as: [header, article, code1, code2, code3, .....]

def parse_text(r):
    article = []
    for p in root.iter('p'):
        article.append(p.text)
    article = ' '.join(article)
    for head in root.iter('headline'):
        header = head.text
    row = [header, article]
    for code in root.iter('code'):
        row.append(code.attrib['code']) 
    return row

# Read through folders recursively and parse only files with subjects I need
# I'll need a function that 'parses' every file to check the subject codes, if they're there then call the parse_text function, otherwise ignore

path = "./RCV1_Unzipped"


#x = [parse_text(root)]


# This recursively traverses the directory tree and appends files from the GCAT category to the list
x = []
for dirName, subdirList, fileList in os.walk(path):
    print('Found directory: %s' % dirName)
    for fname in fileList:
        if fname.endswith('.xml') == True:        
            #print('\t%s' % fname)
            fname = os.path.join(dirName, fname)
            tree = ET.parse(fname)
            root = tree.getroot()
            for code in root.iter('code'):
                if code.attrib == {'code': 'GCAT'}:
                    x.append(parse_text(root))
 
  

# Create a list with the first 3 columns the index, headline and text, then the others are labels
colnames = []
for i in range(54):
    colnames.append("Label_"+str(i-1))
colnames[0] = "Headline"
colnames[1] = "Text"

df = pd.DataFrame(x, columns=colnames) 
df.to_csv('GCAT_RCV1_final.csv') 

colnames_for_filter = colnames[3:]

small_df = df[0:500]


# Read in GCAT_RCV1
#df = pd.read_csv('GCAT_RCV1.csv', encoding = "ISO-8859-1")
#df.drop('Index', inplace=True, axis=1)

'''This is used to construct the boolean to
check in every column whether or not it contains the label of interest'''
def create_boolean_filter(category):
    boolean = []
    for i in range(52):
        boolean.append("(df['Label_"+str(i+1)+"'] == '"+category+"') |")
    boolean = ' '.join(boolean)
    boolean = boolean[:-2]
    return boolean

def create_exclusion_filter(orig_df,category):
    boolean = []
    for i in range(52):
        boolean.append("("+orig_df+"['Label_"+str(i+1)+"'] != '"+category+"') &")
    boolean = ' '.join(boolean)
    boolean = boolean[:-2]
    return boolean

def create_subject_dataframe(df, bool_filter, desired_label):
    subject_df = df[eval(bool_filter)]
    subject_df = subject_df.iloc[:,0:2]
    subject_df['Label'] = desired_label
    subject_df.columns = ['Headline', 'Text', 'Label']
    return subject_df
    
    
def create_subject_dataframe_from_topic(df, desired_label):
    subject_df = df.iloc[:,0:2]
    subject_df['Label'] = desired_label
    subject_df.columns = ['Headline', 'Text', 'Label']
    return subject_df

#### Check GSCI for double labels - are any articles in health and also science?
science = create_boolean_filter('GSCI')

science_full = df[eval(science)]

# Exclude articles which also have the health tag
exclude_health = create_exclusion_filter('science_full','GHEA')

science_without_health = science_full[eval(exclude_health)]


# Other exclusions - C13	REGULATION/POLICY
exclude_C13 = create_exclusion_filter('science_without_health','C13')

science_without_C13 = science_without_health[eval(exclude_C13)]

# C22	NEW PRODUCTS/SERVICES
exclude_C22 = create_exclusion_filter('science_without_C13','C22')

science_without_C22 = science_without_C13[eval(exclude_C22)]

# C23	RESEARCH/DEVELOPMENT
exclude_C23 = create_exclusion_filter('science_without_C22','C23')

science_without_C23 = science_without_C22[eval(exclude_C23)]

GSCI_only = create_subject_dataframe_from_topic(science_without_C23,0)
GSCI_only = GSCI_only.drop_duplicates()


#### repeat for health

health = create_boolean_filter('GHEA')
health_full = df[eval(health)]

exclude_science = create_exclusion_filter('health_full','GSCI')

health_without_science = health_full[eval(exclude_science)]

# Other exclusions - C13	REGULATION/POLICY
exclude_C13 = create_exclusion_filter('health_without_science','C13')

health_without_C13 = health_without_science[eval(exclude_C13)]

# C22	NEW PRODUCTS/SERVICES
exclude_C22 = create_exclusion_filter('health_without_C13','C22')

health_without_C22 = health_without_C13[eval(exclude_C13)]

# C23	RESEARCH/DEVELOPMENT
exclude_C23 = create_exclusion_filter('health_without_C22','C23')

health_without_C23 = health_without_C22[eval(exclude_C23)]


GHEA_only = create_subject_dataframe_from_topic(health_without_C23,1)
GHEA_only = GHEA_only.drop_duplicates()
# Reduce to the size of science dataframe
GHEA_only = GHEA_only[0:1556]


'''
# Defence (8810, 3)
GDEF = create_subject_dataframe(df,create_boolean_filter('GDEF'),0)
 
# Diplomacy (37398, 3)
GDIP = create_subject_dataframe(df, create_boolean_filter('GDIP'),1) 

# CRIME, LAW ENFORCEMENT (31662, 3)
GCRIM = create_subject_dataframe(df, create_boolean_filter('GCRIM'),1) 

# GVIO - WAR, CIVIL WAR (32208, 3)
GVIO = create_subject_dataframe(df, create_boolean_filter('GVIO'),1)

# GENV - (6065, 3)
GENV = create_subject_dataframe(df, create_boolean_filter('GENV'),1)

# Welfare 
GWELF = create_subject_dataframe(df, create_boolean_filter('GWELF'),1)
 
# Sport 
GSPO = create_subject_dataframe(df,create_boolean_filter('GSPO'),1)

# Tourism
GTOUR = create_subject_dataframe(df,create_boolean_filter('GTOUR'),1)

# Disasters
GDIS = create_subject_dataframe(df,create_boolean_filter('GDIS'),1)

# Science
GSCI = create_subject_dataframe(df,create_boolean_filter('GSCI'),1)

# Health
GHEA = create_subject_dataframe(df,create_boolean_filter('GHEA'),1)

NOT_GDEF = concat_frames(GHEA, GSCI)
NOT_GDEF = concat_frames(NOT_GDEF,GDIS)
NOT_GDEF = concat_frames(NOT_GDEF,GTOUR)
NOT_GDEF = concat_frames(NOT_GDEF,GSPO)
NOT_GDEF = concat_frames(NOT_GDEF,GWELF)
NOT_GDEF = concat_frames(NOT_GDEF,GENV)
NOT_GDEF = concat_frames(NOT_GDEF,GCRIM)


NOT_GDEF = shuffle(NOT_GDEF)
NOT_GDEF = NOT_GDEF[0:8810]
 '''
 

 
 
# Concatenate frames and write to CSV
def write_frames_to_csv(frame1,frame2,desired_csv_name):
    frames = [frame1, frame2]
    concat = pd.concat(frames)
    concat.reset_index(inplace=True,drop=True)
    concat = shuffle(concat)
    return concat.to_csv(desired_csv_name+'.csv')


def concat_frames(frame1,frame2):
    frames = [frame1, frame2]
    concat = pd.concat(frames)
    concat.reset_index(inplace=True,drop=True)
    concat = shuffle(concat)
    return concat


# Make classes balanced
'''GDIP = GDIP[0:8810]

GDEF_GDIP = concat_frames(GDIP, GDEF)

GDEF_GDIP['Full_Text'] = GDEF_GDIP['Headline'].map(str) + " " + GDEF_GDIP['Text']
GDEF_GDIP.to_csv('GDEF_GDIP.csv')

'''
GENV_GDEF = pd.concat(frames)
GENV_GDEF.reset_index(inplace=True,drop=True)
GENV_GDEF.to_csv('GENV_GDEF.csv')
'''

GDIS_GENV.to_csv('GDIS_GENV.csv')
'''



GDEF_vs_NOT = concat_frames(GDEF, NOT_GDEF)
GDEF_vs_NOT['Full_Text'] = GDEF_vs_NOT['Headline'].map(str) + " " + GDEF_vs_NOT['Text']


#Make classes balanced
# Health is 1 Science is zero - This matches the application
GHEA_only = GHEA_only[0:1787]

GSCI_GHEA = concat_frames(GHEA_only, GSCI_only)
GSCI_GHEA['Full_Text'] = GSCI_GHEA['Headline'].map(str) + " " + GSCI_GHEA['Text']
GSCI_GHEA.to_csv('GSCI_GHEA_final.csv')
GSCI_GHEA = shuffle(GSCI_GHEA)
GSCI_GHEA.reset_index(inplace=True, drop=True)

###### Topic comparison #####
# Trained on 30 random instances, performance on 500 unseen instances

stopwords_lewis = pd.read_csv('David_Lewis_stopwords.csv')
stopwords_lewis = stopwords_lewis['Stopwords'].tolist()



import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('english')

for i in stop:
    stopwords_lewis.append(i)
    

'''
GDEF_GDIP = shuffle(GDEF_GDIP)

GSCI_GHEA = shuffle(GSCI_GHEA)

GDEF_vs_NOT = shuffle(GDEF_vs_NOT)
'''

# Trained on 10 instances
train_set = GSCI_GHEA[0:10]
# Trained on 30 instances
train_set = GSCI_GHEA[30:60]
train_set['Label'].value_counts()
train_set.to_csv('GSCI_GHEA_train_set_10_model_selection.csv')

test_set = GSCI_GHEA[1000:1500]
test_set.reset_index(inplace=True, drop=True)
test_set.Label.value_counts()
test_set.to_csv('GSCI_GHEA_test_set_final.csv')



vectorizer = TfidfVectorizer(stop_words=stopwords_lewis)
#vectorizer = CountVectorizer(stop_words=stopwords_lewis)

X = vectorizer.fit_transform(train_set['Full_Text'])
y = train_set['Label']


##### Model selection #####
# Define 3 folds (fairly small dataset)

kf = cv.StratifiedKFold(y, n_folds=3, shuffle=True)


# Determine 'optimal' value of C by cross-validation using AUC scoring
# (sklearn uses L2 regularisation by default)
Cs = np.logspace(-4, 4, 10)
gs = grid_search.GridSearchCV(
    estimator=SGDClassifier(loss='log'),
    param_grid={'alpha': Cs},
    scoring='accuracy',
    cv=kf
)
gs.fit(X, y)

gs.best_score_
gs.best_estimator_
gs.grid_scores_

#### Predictions on test set
test_X = vectorizer.transform(test_set['Full_Text'])

predictions = gs.best_estimator_.predict(test_X)

y_test = test_set['Label']

test_accuracy = y_test == predictions
test_accuracy.value_counts()
sum(test_accuracy == True) / len(test_accuracy)


##### Save science vs. health #####
train_set = train_set[0:2]



###### ####### ####### #######
#### Save training set to SQLite
###### ####### ####### #######


conn = sqlite3.connect('RCV1_train.sqlite')
c = conn.cursor()
train_set.to_sql('RCV1_training_set', conn, index_label='index')
conn.commit()
conn.close()

'''
### Read in GDEF vs rest
conn = sqlite3.connect('RCV1_train.sqlite')
c = conn.cursor()
train_set = pd.read_sql("SELECT * FROM RCV1_training_set;",conn)
conn.close()
'''

vectorizer = TfidfVectorizer(stop_words=stopwords_lewis)
X = vectorizer.fit_transform(train_set['Full_Text'])
y = train_set.Label

####### ####### ####### ####### 
# Classifier
####### ####### ####### ####### 
classes = np.array([0,1])
#clf = SGDClassifier(loss='log', learning_rate='constant',eta0=0.5)
clf = SGDClassifier(loss='log', alpha=0.0059948425031894088)
#clf = SGDClassifier()
#clf = LogisticRegression()
# Fit model
#clf.partial_fit(X,RCV1_train_y.values.ravel(), classes=classes)
clf.fit(X,y.values.ravel())
len(clf.coef_[0])


####### Test set 
test_set['indexID'] = test_set.index


test_set['prediction'] = clf.predict(test_X)
test_set['predicted_labels'] = test_set['prediction'].map({0:'Science', 1:'Health'})
test_set['class_proba'] = clf.predict_proba(test_X)[:,1]
test_set = test_set[['indexID', 'Label', 'Headline', 'Text', 'Full_Text', 'prediction', 'predicted_labels', 'class_proba']]


test_initial_accuracy = test_set['prediction'] == test_set['Label']
pd.Series(test_initial_accuracy).value_counts()



##### Test set for science vs health
test_X = vectorizer.transform(test_set['Full_Text'])

test_set['indexID'] = test_set.index


test_set['prediction'] = clf.predict(test_X)
test_set['predicted_labels'] = test_set['prediction'].map({0:'Science', 1:'Health'})
test_set['class_proba'] = clf.predict_proba(test_X)[:,1]
test_set = test_set[['indexID', 'Label', 'Headline', 'Text', 'Full_Text', 'prediction', 'predicted_labels', 'class_proba']]





####### ####### ####### ####### 
# Save test set to SQLite
####### ####### ####### ####### 
conn = sqlite3.connect('RCV1.sqlite')
c = conn.cursor()
test_set.to_sql('RCV1_test_X', conn, index_label='index')
conn.commit()
conn.close()


# Pickle classifier


dest = os.path.join('Flask-Application-Reuters20', 'pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)
#pickle.dump(stop,open(os.path.join(dest, 'stopwords.pkl'), 'wb'),protocol=4)
pickle.dump(clf,open(os.path.join(dest, 'RCV1_log_reg_GSCI_GHEA.pkl'), 'wb'), protocol=4)


# Pickle the vectorizer
dest = os.path.join('Flask-Application-Reuters20', 'pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)
pickle.dump(vectorizer,open(os.path.join(dest, 'TF_IDF_vect.pkl'), 'wb'), protocol=4)

# Pickle stopwords list
dest = os.path.join('Flask-Application-Reuters20', 'pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)
#pickle.dump(stop,open(os.path.join(dest, 'stopwords.pkl'), 'wb'),protocol=4)
pickle.dump(stopwords_lewis,open(os.path.join(dest, 'stopwords.pkl'), 'wb'), protocol=4)







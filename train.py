
# coding: utf-8

# In[54]:


from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd  
import numpy as np
import pickle
import codecs
import spacy
import glob
import re

all_files = glob.glob("/corpus/groupe*.txt")
df = pd.DataFrame()
for file_ in all_files: 
    list_of_lists = []
    with open(file_, 'rU') as f:
        for line in f:
            tweet = ' '.join(line.split(")")[1:])
            id = line.split(")")[0].split(",")[0][1:]
            sentiment = line.split(")")[0].split(",")[1]
            try:
                group = line.split(")")[0].split(",")[2]
            except:
                group = ''
            list_of_lists.append([id, sentiment, group, tweet])
    next_df = pd.DataFrame(list_of_lists, columns=['Id','Sentiment', 'Group', 'Tweet'])
    df = df.append(next_df)
df = df.drop_duplicates()
df = df.reset_index(drop=True)
df.info()


# In[55]:


print(df.Sentiment.value_counts())


# In[56]:


contraction_mapping = {
    "gôche": "gauche",
    "gauchos": "gauche",
    "réac": "réaction",
    "co": "companie",
    "cie": "companie",
    "and": "et",
    "but": "mais",
    "ui": "oui",
    "@macron": "Macron",
    "@EmmanuelMacron": "Emmanuel Macron",
    "GJ": "gilet jaune",
    " 1 ": " un ",
    "PR": "président",
    "RT": "retweet",
    "LREM": "la république en marche",
    "LAREM": "la république en marche",
    "#LREM": "la république en marche",
    "#LAREM": "la république en marche",
    "telma": "tellement",
    "ctr": "contre",
    "5e": "cinquieme",
    "foute": "foutre",
    "my god": "mon dieu",
    "nn": "non",
    "b8en": "bien",
    "good": "bien",
    "bad": "mauvais",
    "lui": "il"
}


# In[57]:


nlp = spacy.load('fr')

def tweet_cleaner(text):
    decoded = str(text)
    apostrophe_handled = re.sub("’", "'", decoded)
    expanded = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in apostrophe_handled.split(" ")])
    parsed = nlp(expanded)
    final_tokens = []
    for t in parsed:
        if t.is_punct or t.is_space or t.like_num or t.like_url or str(t).startswith('@'):
            pass
        else:
            if t.lemma_ == '-PRON-':
                final_tokens.append(str(t))
            else:
                sc_removed = re.sub("[^a-zA-Zéèêùûàâœçî]", '', str(t.lemma_))
                if len(sc_removed) > 1:
                    final_tokens.append(sc_removed)
    joined = ' '.join(final_tokens)
    spell_corrected = re.sub(r'(.)\1+', r'\1\1', joined)
    return spell_corrected


# In[58]:


pd.set_option('display.max_colwidth', -1)
print(df.Tweet[131:135])


# In[59]:


print([tweet_cleaner(t) for t in df.Tweet[131:135]])


# In[60]:


df['Clean_tweet'] = [tweet_cleaner(t) for t in df.Tweet]


# In[61]:


df.head()


# In[62]:


#Get stopwords from file
with open('./stop_words_fr.txt') as f:
    stopwords = f.read().splitlines()


# In[63]:


X = df.Clean_tweet.values
y = df.Sentiment.values

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# split train & test
training_X, test_X, training_y, test_y = train_test_split(X, y, test_size = .3, random_state=42)

sm = SMOTE(random_state=42)
X_train_s, y_train_s = sm.fit_sample(training_X, training_y)

X_train, X_test, y_train, y_test = train_test_split(X_train_s, y_train_s, test_size = .3, random_state=42)

# model tuning & validation
cv = StratifiedKFold()


# In[64]:


rf_grid = {  
     #'max_depth': list(range(10,100,10)),
     #'max_features': ['auto', 'sqrt'],
     #'min_samples_leaf': [1, 2, 4],
     #'min_samples_split': [2, 5, 10],
     'n_estimators': list(range(100,500,100))
}
                  
rf_clf = RandomForestClassifier(n_estimators=35, max_depth=8, min_samples_split=4,n_jobs=-1)

best_rf_clf = GridSearchCV(rf_clf, rf_grid, verbose=2,cv=cv)
best_rf_clf.fit(X_train, y_train)
rf_preds = best_rf_clf.predict(test_X)

print("Random Forest")
print(best_rf_clf.best_params_)
print(classification_report(test_y, rf_preds))
print("Accuracy", accuracy_score(test_y, rf_preds))


# In[65]:


lr_grid = {
    "C": np.logspace(-4,4,20), 
    "penalty": ["l1","l2"] # l1 lasso l2 ridge
}

lr_clf = LogisticRegression()

best_lr_clf = GridSearchCV(lr_clf, lr_grid, verbose=2,cv=cv)
best_lr_clf.fit(X_train, y_train)
lr_preds = best_lr_clf.predict(test_X)

print("Logistic Regression")
print(best_lr_clf.best_params_)
print(classification_report(test_y, lr_preds))
print("Accuracy", accuracy_score(test_y, lr_preds))


# In[66]:


# Save rf classifier to a file
save_classifier = open("rf_classifier.pickle", 'wb')
pickle.dump(best_rf_clf, save_classifier)
save_classifier.close()

# Save lr classifier to a file
save_classifier = open("lr_classifier.pickle", 'wb')
pickle.dump(best_lr_clf, save_classifier)
save_classifier.close()


# In[67]:


# Retrieve the saved file and uplaod it to an object
vec = open("rf_classifier.pickle", 'rb')
rf_clf = pickle.load(vec)
vec.close()

# Retrieve the saved file and uplaod it to an object
vec = open("lr_classifier.pickle", 'rb')
lr_clf = pickle.load(vec)
vec.close()


# In[68]:


#Example of oversamling with SMOTE

sent1 = "emmanuel macron etre bon"
sent2 = "macron etre un monstre"
sent3 = "macron sale merde"
sent4 = "comme il etre mauvais ce macron"
sent5 = "il avoir bien aimer macron"

testing_text = pd.Series([sent1, sent2, sent3, sent4, sent5])
testing_target = pd.Series([1,0,0,0,1])

tv = TfidfVectorizer(stop_words=None, max_features=100000)
testing_tfidf = tv.fit_transform(testing_text)

smt = SMOTE(random_state=777, k_neighbors=1)
X_SMOTE, y_SMOTE = smt.fit_sample(testing_tfidf, testing_target)
df_topredict = pd.DataFrame(X_SMOTE.todense(), columns=tv.get_feature_names())
df_topredict


# In[69]:


print(y_SMOTE)


# In[70]:


pd.DataFrame(test_X.todense(), columns=vectorizer.get_feature_names()).info()


# In[71]:


container_df = pd.DataFrame(0.0, index=np.arange(len(X_SMOTE.todense())), columns=vectorizer.get_feature_names())


# In[72]:


for column in df_topredict:
    if column in vectorizer.get_feature_names():
        container_df[column] = df_topredict[column]


# In[73]:


preds = best_lr_clf.predict(container_df)
preds


# In[74]:


preds = best_rf_clf.predict(container_df)
preds


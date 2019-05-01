
# coding: utf-8

# In[36]:


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

all_files = glob.glob("./corpus/groupe*.txt")
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


# In[37]:


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


# In[38]:


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


# In[39]:


df['Clean_tweet'] = [tweet_cleaner(t) for t in df.Tweet]


# In[40]:


#Get stopwords from file
with open('./stop_words_fr.txt') as f:
    stopwords = f.read().splitlines()


# In[41]:


X = df.Clean_tweet.values
y = df.Sentiment.values

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)


# In[46]:


def get_predictions(file):    
    #Convert the file corpus to a dataframe
    list_of_lists = []
    with open(file, 'rU') as f:
        for line in f:
            tweet = ' '.join(line.split(")")[1:])
            id = line.split(")")[0].split(",")[0][1:]
            sentiment = line.split(")")[0].split(",")[1]
            try:
                group = line.split(")")[0].split(",")[2]
            except:
                group = ''
            list_of_lists.append([id, sentiment, group, tweet])
    df = pd.DataFrame(list_of_lists, columns=['Id','Sentiment', ' ', 'Tweet'])
    print(df.info())
    
    #Clean and transform the data
    df['Clean_tweet'] = [tweet_cleaner(t) for t in df.Tweet]
    
    X = df.Clean_tweet.values
    y = df.Sentiment.values
    
    tv = TfidfVectorizer()
    X = tv.fit_transform(X)
    
    #Use the saved classifier and evaluate the predictions
    vec = open("rf_classifier.pickle", 'rb')
    clf = pickle.load(vec)
    vec.close()
    
    #Use a container for the dataframe to make the prediction
    container_df = pd.DataFrame(0.0, index=np.arange(len(X.todense())), columns=vectorizer.get_feature_names())
    df_topredict = pd.DataFrame(X.todense(), columns=tv.get_feature_names())
    for column in df_topredict:
        if column in vectorizer.get_feature_names():
            container_df[column] = df_topredict[column]

    preds = clf.predict(container_df)
    df['Predicted_sentiment'] = preds
    print(preds)
    
    annoted_file_name = "Annoted_" + file
    text_file = open(annoted_file_name, "w")
    for index, row in df.iterrows():
        line = '(' + row['Id'] + ', ' + row['Predicted_sentiment'] + ') ' + row['Tweet']
        text_file.write(line)
    text_file.close()
    print("Annoted file created with success: ", annoted_file_name)


# In[47]:


get_predictions('TweetsAboutMacron.txt')


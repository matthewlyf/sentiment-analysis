# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 10:43:17 2020

@author: 16478
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
df1 = pd.read_csv("C:/Users/16478/Desktop/sentiment-analysis-master/data/raw/Training_full_label.csv")

##############################SST
##sst_data = pd.read_csv("C:/Users/16478/Desktop/sentiment-analysis-master/data/testing/stanfordSentimentTreebank/stanfordSentimentTreebank/original_rt_snippets.txt")
#sst_labels =pd.read_csv("C:/Users/16478/Desktop/sentiment-analysis-master/data/testing/stanfordSentimentTreebank/stanfordSentimentTreebank/sentiment_labels.txt", sep= "|", header = 0)
############
APPOSTOPHES = {
"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"in vitro": "invitro",
"in vivo": "invivo"
}

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords 
from stop_words import get_stop_words
stop_words = list(get_stop_words('en'))         #About 900 stopwords
nltk_words = list(stopwords.words('english')) #About 150 stopwords
stop_words.extend(nltk_words)
stop_words = set(stopwords.words('english')) 

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree.export import export_text
import re
import nltk



preprocessed_abstract = []
for n in range(0, len(df1)):
    print(n)
    abstract_process = str(df1['message'][n])   
    abstract_process = re.sub(r"[^'/. A-Za-z]",' ', abstract_process)
    abstract_process = re.sub(r"[/]",' ', abstract_process)
    abstract_process = re.sub(r"[.]",' ', abstract_process)
    abstract_process = re.sub(r'[\d]','', abstract_process)
    abstract_process = abstract_process.lower()
    abstract_process = abstract_process.strip()
    word_tokens = abstract_process.split()#splits words from comments into list
    #Step 6: Map words to contracted word dictionary and substitute them       
    word_tokens = [APPOSTOPHES[word] if word in APPOSTOPHES else word for word in  word_tokens]
    '''remove any items in list that is empty as it 
    causes error to pos_tag'''
    word_tokens = [w for w in word_tokens if len(w) > 0]
    #Step 7: Parts of Speech tagging of each word
    from nltk.stem import WordNetLemmatizer
    wnl = WordNetLemmatizer()
    new_word_token_list = []
    new_word_token = nltk.pos_tag(word_tokens)
    new_word_token_list.append(new_word_token)
    [new_word_token_list] = new_word_token_list
    #Step 8: Lemmatize Tokenized Text
    post_lemm = []       
    for word,tag in new_word_token_list:
        wntag = tag[0].lower()
        wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
        if not wntag:
            lemma = word
            post_lemm.append(lemma)
        else:
            lemma = wnl.lemmatize(word, wntag)
            post_lemm.append(lemma)
    filtered_sentence = [] 
    for w in post_lemm: 
        if w not in stop_words: 
            filtered_sentence.append(w)
    filtered_sentence = [i for i in filtered_sentence if len(i) > 1]
    filtered_sentence = ' '.join(filtered_sentence)       
    filtered_sentence = re.sub(r"[^\w\s]",'', filtered_sentence)
    '''for some reason we need to tokenize, identify words greater than
    1 character, and join a second time to get cleaner data'''
    filtered_sentence = filtered_sentence.split()
    filtered_sentence = [i for i in filtered_sentence if len(i) > 1]       
    filtered_sentence = ' '.join(filtered_sentence)        
    preprocessed_abstract.append(filtered_sentence)










preprocessed_abstract = pd.Series(preprocessed_abstract)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(preprocessed_abstract, df1['sentiment'], test_size=0.33, random_state=42)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
X_test_counts = count_vect.fit_transform(X_test)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

x_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)
x_test_tfidf.shape



#Naive Bayes Model
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, y_train)

#Support vector machine Model
from sklearn.linear_model import SGDClassifier
svm = SGDClassifier().fit(X_train_tfidf, y_train)


#C4.5 Decision Tree Model
from sklearn import tree
tree = tree.DecisionTreeClassifier().fit(X_train_tfidf, y_train)

#If you want to predict use this:
import numpy as np
X_new_counts = count_vect.transform(X_test)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)
np.mean(predicted == y_test)
########################################
from sklearn.metrics import classification_report, accuracy_score, make_scorer

def classification_report_with_accuracy_score(y_true, y_pred):

    print (classification_report(y_true, y_pred)) # print classification report
    return accuracy_score(y_true, y_pred) # return accuracy score



import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#Cross Validation using 10 folds

kf = KFold(n_splits=10)



print("Naive Bayes Results")
results_kfold = cross_val_score(clf, x_test_tfidf, y_test, cv=kf, scoring=make_scorer(classification_report_with_accuracy_score))
print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0)) 
print(results_kfold)



print("Support Vector Machine Results")
svm_results_kfold = cross_val_score(svm, x_test_tfidf, y_test, cv=kf,  scoring=make_scorer(classification_report_with_accuracy_score))
print("Accuracy: %.2f%%" % (svm_results_kfold.mean()*100.0)) 
print(svm_results_kfold)

print("Decision tree results")
tree_results_kfold = cross_val_score(tree, x_test_tfidf, y_test, cv=kf,scoring=make_scorer(classification_report_with_accuracy_score))
print("Accuracy: %.2f%%" % (tree_results_kfold.mean()*100.0)) 
print(tree_results_kfold)

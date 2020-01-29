# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 13:00:43 2020

@author: Matthew Li Yuen Fong
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 16:31:56 2020

@author: Matthew Li Yuen Fong
"""
import pandas as pd
import pymongo
import pyspark
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict
frequency = defaultdict(int)
new_frequency = defaultdict(int)
import pprint
from gensim import corpora
from textblob import TextBlob

client = pymongo.MongoClient(<'Server+URL'>)
print(client.list_database_names())
mydb = client["leagueoflegends_subreddit"]
mycol2 = mydb["dictionary_raw"]
mycol = mydb["dictionary"] #should be processed data
data = pd.DataFrame(list(mycol.find()))
data_raw = pd.DataFrame(list(mycol2.find()))
#print(data)

#print(data['message'])


#ngrams(sentence.split(), n)
#totals = data.message.ngrams.split(),n.stack().value_counts()


#print (list(ngram_msg))



'''this is a script to query and search for relevant comments using tf-idf models'''
processed_messages = []

for messages in range(0, len(data)):
    word_tokens_message = word_tokenize(data['message'][messages])
    for text in word_tokens_message:
        frequency[text] += 1 #this works, frequency counter



    # Only keep words that appear more than once

    processed_corpus = [text for text in word_tokens_message if frequency[text] >2]
    for text in processed_corpus:
        new_frequency[text] += 1
    processed_messages.append(processed_corpus)
   # print(processed_messages)
   
###############################################################################
#########################Create dataframe of word frequencies##################
###############################################################################

wordfrequency = pd.DataFrame([new_frequency])
wordfrequency= wordfrequency.T
wordfrequency= wordfrequency.reset_index()
wordfrequency= wordfrequency.rename(columns= {"index" : "word", 0 : "Count"})
wordfrequency.to_csv()    

###############################################################################
###############################################################################
###############################################################################


#creates list of unique words
dictionary = corpora.Dictionary(processed_messages)
print(dictionary)
#prints list of unique words
#pprint.pprint(dictionary.token2id)

#maps each entry in message to dictionary
bow_corpus = [dictionary.doc2bow(text) for text in processed_messages]
#pprint.pprint(bow_corpus)


from gensim import models

# train the model
tfidf = models.TfidfModel(bow_corpus)

# transform the [input] string and see if it appears in dictionary
#words = "pretty simple find".lower().split()
#print(tfidf[dictionary.doc2bow(words)])

#find input string query and return docs that are relevant, REMEMBER TO UPDATE NUMBER OF FEATURES
from gensim import similarities
index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=7014) #update num_features
query_document = 'tsm'.split()
query_bow = dictionary.doc2bow(query_document)
sims = index[tfidf[query_bow]]
#print(list(enumerate(sims)))

#returns sorted list by index of your query
doc_no = []
score_lst = []
msg = []
column_names = ['document_number','score','message']
query_list = pd.DataFrame(columns = column_names)

#returns top documents containing query and puts it in a dataframe.  This dataframe can be used for sentiment analysis
for document_number, score in sorted(enumerate(sims), key=lambda x: x[1], reverse=True):
    #print(document_number, score, data['message'][document_number])
    if score > 0:
        doc_no.append(document_number)
        score_lst.append(score)
        msg.append(data['message'][document_number])#this is the result used in sentiment analysis
    else:
        pass
result = pd.concat([pd.Series(doc_no, name = "doc_no"), pd.Series(score_lst, name = "score"), pd.Series(msg, name = 'msg')], axis=1)
print(result)    
'''end of script'''

'''Sentiment analysis script using VADER'''
#vader needs source comments, not preprocessed, comments need to be in a single list
#positive sentiment : (compound score >= 0.05)
#neutral sentiment : (compound score > -0.05) and (compound score < 0.05)
#negative sentiment : (compound score <= -0.05)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()

def print_sentiment_scores(sentence):
    sentiment_dict  = analyser.polarity_scores(sentence)
    #pprint.pprint("{:-<40} {}".format(sentence, str(snt)))
    print("Overall sentiment dictionary is : ", sentiment_dict)
    #print(result['msg'][msgs])
    print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative") 
    print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral") 
    print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive") 
  
    print("Sentence Overall Rated As", end = " ") 
  
    # decide sentiment as positive, negative and neutral 
    if sentiment_dict['compound'] >= 0.05 : 
        print("Positive") 
  
    elif sentiment_dict['compound'] <= - 0.05 : 
        print("Negative") 
  
    else : 
        print("Neutral") 
    print("")

print("Overall consensus of query")
single_msg_list = result['msg'].values.tolist()
single_msg_list = ' '.join(single_msg_list)
print_sentiment_scores(single_msg_list)
print("-----")

#classify all msgs individually using VADER for returned query#
for msgs in range(0,len(result)):
    
    pprint.pprint(print_sentiment_scores(result['msg'][msgs]))


##################################################################################
##############CREATES TRAINING DATA FOR MACHINE LEARNING##########################
##################################################################################
sentiment_label = []
message_full = []
compound_score_full = []
process_msg = []
for full_text_words in range(0, len(data_raw)):
    sent_train  = analyser.polarity_scores(data_raw['message'][full_text_words])
    if sent_train['compound'] >= 0.05 : 
        sentiment_label.append("positive") 
        message_full.append(data_raw['message'][full_text_words])
        process_msg.append(data['message'][full_text_words])
        compound_score_full.append(sent_train['compound'])
    elif sent_train['compound'] <= - 0.05 : 
        sentiment_label.append("negative") 
        message_full.append(data_raw['message'][full_text_words])
        process_msg.append(data['message'][full_text_words])
        compound_score_full.append(sent_train['compound'])
        
    else : 
        sentiment_label.append("neutral")  
        message_full.append(data_raw['message'][full_text_words])
        process_msg.append(data['message'][full_text_words])
        compound_score_full.append(sent_train['compound'])
train_data = pd.concat([pd.Series(sentiment_label, name = "sentiment_label"),pd.Series(process_msg, name = "process_msg"), pd.Series(message_full, name = "msg"),pd.Series(compound_score_full, name = "compound score")], axis=1)
train_data.to_csv()        
##################################################################################
##################################################################################
##################################################################################




'''Textblob sentiment'''
'''Polarity is float which lies in the range of [-1,1] 
where 1 means positive statement and -1 means a negative statement. 
Subjective sentences generally refer to personal opinion, emotion or 
judgment whereas objective refers to factual information. 
Subjectivity is also a float which lies in the range of [0,1].'''


TB_text = TextBlob(single_msg_list)
TB_text = TB_text.correct()
print(TB_text.sentiment)

#classify all msgs individually using TextBlob for returned query#
for msgs in range(0,len(result)):
    TB_text = TextBlob(result['msg'][msgs])
    TB_text = TB_text.correct()
    print(TB_text.sentiment)
    
#for msgs in range(0, len(result)):
#    print_sentiment_scores(result['msg'][msgs])


'''
totals = data.message.str.split(expand=True).stack().value_counts()
'''
#totals  = pd.DataFrame(totals)

'''
sc = pyspark.SparkContext('local[*]')
df = sc.createDataFrame(data)
print(df.count())
'''

#token
'''
token_message = []
for n in range(0, len(data['message'])):
    tk = word_tokenize(data['message'][n])
    tk = nltk.bigrams(tk) 
    
    tk = list(tk)
    
    for a in range(0,len(tk)):
        tk[a] = ' '.join(tk[a])
    token_message.append(tk)
data2 = pd.DataFrame(token_message)
'''
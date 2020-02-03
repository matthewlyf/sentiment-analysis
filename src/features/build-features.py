#Initialise Database + Reddit
import os
from dotenv import load_dotenv, find_dotenv

#Step 1: find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()

#Step 2: load up the entries as environment variables
load_dotenv(dotenv_path, override=True)

processed_path =  os.getenv("processed_path")
raw_path =  os.getenv("raw_path")

'''takes dataframe with tokenized messages as variable input,
if scrape_reddit() was run, it can take variable all_col_toke'''
#Step 0: Prepare libraries, database, and dataframes
import pandas as pd
from collections import defaultdict
frequency = defaultdict(int)
data = pd.read_csv(raw_path+"raw.csv")
data_token = pd.read_csv(processed_path+"processed+tokenized.csv")
processed_messages = []
#.apply(eval) required as loading lists in read_csv doens't work)
data_token['message'] = data_token['message'].apply(eval)
#Step 1: Count words and record frequency
for messages in data_token['message']:
    for text in messages:
        frequency[text] += 1 
#Step 2: Assuming words appearing once are unimportant, removed words appearing
#less than 2 times
    processed_corpus = [text for text in messages if frequency[text] >2]
#Step 3: Take filtered messages and create final list
    processed_messages.append(processed_corpus)
#Step 4: Convert final frequncy data into a dataframe   
wordfrequency = pd.DataFrame([frequency])
wordfrequency= wordfrequency.T
wordfrequency= wordfrequency.reset_index()
wordfrequency = wordfrequency.rename(columns= {"index" : "word", 0 : "Count"})
wordfrequency.to_csv(raw_path+"wordfrequency.csv")

def tfidf(query, processed_messages):
    #Step 1: Create corpus from processed messages(this can be accessed from 
    #processed_messages)    
    import gensim as gm
    import pandas as pd
    import pprint
    dictionary = gm.corpora.Dictionary(processed_messages)
    bow_corpus = [dictionary.doc2bow(text) for text in processed_messages]
    #Step 2: Create TF-IDF model from corpus
    tfidf = gm.models.TfidfModel(bow_corpus)
    #TODO num_features is the value of unique tokens from dictionary, need
    #to extract that value
    index = gm.similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=7014) 
    query_document = query.split()
    query_bow = dictionary.doc2bow(query_document)
    sims = index[tfidf[query_bow]]
    doc_no = []
    score_lst = []
    msg = []   
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
    pprint.pprint(result.head())    
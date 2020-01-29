
import pprint
import re
from stop_words import get_stop_words
import pymongo
import praw
import pandas as pd
import datetime as dt
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english')) 
from collections import defaultdict
frequency = defaultdict(int)
from textblob import TextBlob
client = pymongo.MongoClient(<'Server+URL'>)
print(client.list_database_names())
client.drop_database('leagueoflegends_subreddit')
mydb = client["leagueoflegends_subreddit"]
#mydb["lol_overview"].drop()
mycol = mydb["lol_overview"]
mycol2 = mydb["leagueoflegends_pre"]
stop_words = list(get_stop_words('en'))         #About 900 stopwords
nltk_words = list(stopwords.words('english')) #About 150 stopwords
stop_words.extend(nltk_words)
nltk.download('averaged_perceptron_tagger')
############################dictionary################################
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
"u": "you",
"'s" :"",
"/s": "sarcasm",
"/u": "user "}
############################################
from datetime import datetime
 
 
reddit = praw.Reddit(client_id='client_id', \
                     client_secret='client_secret', \
                     user_agent='comment stream', \
                     username='username', \
                     password='password')

###############################################################################
###############################Get TimeStamp (Unused)########################## 
###############################################################################
def get_yyyy_mm_dd_from_utc(dt):
    date = datetime.utcfromtimestamp(dt)
    
    return str(date.year) + "-" + str(date.month) + "-" + str(date.day)
############################################################################### 
############################################################################### 
############################################################################### 

subreddit = reddit.subreddit('leagueoflegends') 
top_subreddit = subreddit.hot(limit=20)
for submission in top_subreddit:
###############################################################################
#######################Create Collections in MongoDB###########################   
###############################################################################

    all_col = mydb['dictionary']
    all_col_raw = mydb['dictionary_raw']    
    topics_dict = { "title": submission.title , "score":submission.score, 
                   "id":submission.id, "url":submission.url, 
                   "comms_num":submission.num_comments , 
                   "created":get_yyyy_mm_dd_from_utc(submission.created) , 
                   "body":submission.selftext, 
                   "z_comments":[]}
 
    mycol.insert_one(topics_dict) 
    print("1")
    title = str(submission.title)
    title = title[0:20]
    print("2")
    new_col = mydb[title]

    print("3")
    
###############################################################################
##########################Scrape Reddit for Comments###########################
###############################################################################
    
    submission.comments.replace_more(limit=None)
    for comment in submission.comments.list():
        mydict1 = {"message":comment.body}
        raw_comment = {"message":comment.body}
        all_col_raw.insert_one(raw_comment)
        new_col.insert_one(mydict1)
        
###############################################################################
#############################PreProcess Comments###############################
###############################################################################   
        comments_stream = comment.body
        comments_stream = re.sub(r"[^'. A-Za-z]",'', comments_stream)
        comments_stream = re.sub(r"[.]",' ', comments_stream)
        comments_stream = re.sub(r'[\d]','', comments_stream)
        comments_stream = comments_stream.lower()
        comments_stream = comments_stream.strip() #remove white space
######################Tokenize Comments########################################        
        word_tokens = comments_stream.split()#splits words from comments into list
        
###########################Convert listed Words################################
        word_tokens = [APPOSTOPHES[word] if word in APPOSTOPHES else word for word in  word_tokens]
        
        #remove any items in list that is empty as it causes error to pos_tag
        word_tokens = [w for w in word_tokens if len(w) > 0]

        from nltk.stem import WordNetLemmatizer
        wnl = WordNetLemmatizer()
        new_word_token_list = []
        new_word_token = nltk.pos_tag(word_tokens)
        new_word_token_list.append(new_word_token)
        [new_word_token_list] = new_word_token_list
########################### Lemmatize Tokenized Text ##########################
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
                if word != lemma:
                    print("i've been shortened from ", word, lemma)
                else:
                    pass
########################### Remove StopWords###################################        
        filtered_sentence = [] 
        for w in post_lemm: 
            if w not in stop_words: 
                filtered_sentence.append(w) #appends remaining words into new list
        
        filtered_sentence = [i for i in filtered_sentence if len(i) > 1]
        filtered_sentence = ' '.join(filtered_sentence)#removes, and joins list witha space
        filtered_sentence = filtered_sentence.lower()
        filtered_sentence = re.sub(r"[^\w\s]",'', filtered_sentence)
        #requires another pass 
        filtered_sentence = filtered_sentence.split()
        filtered_sentence = [i for i in filtered_sentence if len(i) > 1]
        filtered_sentence = ' '.join(filtered_sentence)
        
        mydict2 = {"message":filtered_sentence}
        all_col.insert_one(mydict2)
    print("Done", submission.title)
        #comment_body =  comment_body + comment.body + "\n"
print("Done scraping")       

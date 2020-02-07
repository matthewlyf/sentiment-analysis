#Initialise Database + Reddit
import os
from dotenv import load_dotenv, find_dotenv

#Step 1: find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()

#Step 2: load up the entries as environment variables
load_dotenv(dotenv_path, override=True)

database_url = os.getenv("DATABASE_URL")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
user_agent = os.getenv("user_agent")
username = os.getenv("r_username")
password = os.getenv("r_password")

#Step3: Prepare reddit and mongodb instance
import pymongo
import praw
  
client = pymongo.MongoClient(database_url)
reddit = praw.Reddit(client_id= REDDIT_CLIENT_ID, 
                     client_secret= REDDIT_CLIENT_SECRET, \
                     user_agent= user_agent, \
                     username= username, \
                     password= password)



#Make-Dataset Script
#TODO: Update subreddit_name with target subreddit
#TODO: Update submission limit for scrape
#TODO: Update file paths for data export
raw_path =  os.getenv("raw_path")
processed_path =  os.getenv("processed_path")
subreddit_name = "leagueoflegends"
submission_limit = 20
#Step 0: Prepare libraries and packages
import pandas as pd
import re
import nltk
from datetime import datetime
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords 
from stop_words import get_stop_words
stop_words = list(get_stop_words('en'))         #About 900 stopwords
nltk_words = list(stopwords.words('english')) #About 150 stopwords
stop_words.extend(nltk_words)
stop_words = set(stopwords.words('english')) 

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

def get_yyyy_mm_dd_from_utc(dt):
    date = datetime.utcfromtimestamp(dt) 
    return str(date)
#Step 1: Prepare mongoDB database
subreddit = reddit.subreddit(subreddit_name)
str_subreddit = subreddit_name+"_subreddit"
str_overview = subreddit_name+"_overview"
mydb = client[str_subreddit]
client.drop_database(str_subreddit)
mydb = client[str_subreddit]
mycol = mydb[str_overview]
#Step 2: Store top "N" submission titles from front page into mongoDB
top_subreddit = subreddit.hot(limit=submission_limit)
for submission in top_subreddit:
    all_col = mydb['dictionary_processed']
    all_col_raw = mydb['dictionary_raw']
    
    all_col_toke = mydb['dictionary_processed+tokenized']
    topics_dict = { "title": submission.title , "score":submission.score, 
                   "id":submission.id, "url":submission.url, 
                   "comms_num":submission.num_comments , 
                   "body":submission.selftext, 
                   "z_comments":[]}    
    mycol.insert_one(topics_dict) 
    title = str(submission.title)
    '''title needs to be sliced due to character limitations'''
    title = title[0:20]
    title = re.sub(r"(\.\.\.)"," ", title)
#    count = 0
#    if title == "":
#        count += 1
#        title = "Error - Empty"+ count
#        new_col = mydb["Error - Empty"]
#    else:
    new_col = mydb[title]
#Step 3: Scrape top "N" submission comments and store in mongoDB
    submission.comments.replace_more(limit=None)
    for comment in submission.comments.list():
        mydict1 = {"message":comment.body}
        raw_comment = {"message":comment.body, "title":str(submission.title), "timestamp": get_yyyy_mm_dd_from_utc(comment.created_utc)}
        all_col_raw.insert_one(raw_comment)
        new_col.insert_one(mydict1)      
#Step 4: Extract comments and remove symbols, spaces, punctation, and digits
#except for <'>
#TODO: revisit digit removal
        comments_stream = comment.body
        comments_stream = re.sub(r"[^'/. A-Za-z]",'', comments_stream)
        comments_stream = re.sub(r"[/]",' ', comments_stream)
        comments_stream = re.sub(r"[.]",' ', comments_stream)
        comments_stream = re.sub(r'[\d]','', comments_stream)
        comments_stream = comments_stream.lower()
        comments_stream = comments_stream.strip() #remove white space
#Step 5: Tokenize comments      
        word_tokens = comments_stream.split()#splits words from comments into list
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
#                    '''this conditional statement can be removed at a 
#                    later point in time'''
#                    if word != lemma:
#                        print("i've been shortened from ", word, lemma)
#                    else:
#                        passS
#Step 9: Remove Stop words and re-join comments     
        filtered_sentence = [] 
        for w in post_lemm: 
            if w not in stop_words: 
                filtered_sentence.append(w)
        filtered_sentence = [i for i in filtered_sentence if len(i) > 1]
        filtered_sentence = ' '.join(filtered_sentence)
        filtered_sentence = filtered_sentence.lower()
        filtered_sentence = re.sub(r"[^\w\s]",'', filtered_sentence)
        '''for some reason we need to tokenize, identify words greater than
        1 character, and join a second time to get cleaner data'''
        filtered_sentence = filtered_sentence.split()
        filtered_sentence = [i for i in filtered_sentence if len(i) > 1]
        token_filtered_sent = {"message":filtered_sentence, "topic": str(submission.title), "timestamp": get_yyyy_mm_dd_from_utc(comment.created_utc)}
        all_col_toke.insert_one(token_filtered_sent)
        filtered_sentence = ' '.join(filtered_sentence)
#Step 10: Store processed messages in mongoDB        
        mydict2 = {"message":filtered_sentence, "topic":str(submission.title), "timestamp": get_yyyy_mm_dd_from_utc(comment.created_utc)}
        all_col.insert_one(mydict2)
    print("Done", submission.title)
        #comment_body =  comment_body + comment.body + "\n"
print("Done scraping")

#Step 11: Export resulting datasets to directories
export_raw = mydb["dictionary_raw"]
export_processed = mydb["dictionary"]
export_tokenized = mydb['dictionary_token']
raw_df = pd.DataFrame(list(export_raw.find()))
processed_df = pd.DataFrame(list(export_processed.find()))
tokenized_df = pd.DataFrame(list(export_tokenized.find()))
tokenized_df.to_csv(processed_path+"processed+tokenized.csv")
raw_df.to_csv(raw_path+"raw.csv")
processed_df.to_csv(processed_path+"processed.csv")

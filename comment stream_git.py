'''
scrpr(subreddit_name, reddit, client)
wrdfrq(all_col_toke)
'''



     
def scrpr(subreddit_name, reddit, client):
    '''takes 3 variables
    subreddit name should be the exact string or variable containing string of
    subreddit name
    
    reddit should be your reddit instance for praw
    client should be your client instance for mongodb'''
#Step 0: Prepare libraries and packages
    import re
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
#Step 1: Prepare mongoDB database
    subreddit = reddit.subreddit(subreddit_name)
    str_subreddit = subreddit_name+"_subreddit"
    str_overview = subreddit_name+"_overview"
    mydb = client[str_subreddit]
    client.drop_database(str_subreddit)
    mydb = client[str_subreddit]
    mycol = mydb[str_overview]
#Step 2: Store top "N" submission titles from front page into mongoDB
    top_subreddit = subreddit.hot(limit=20)
    for submission in top_subreddit:
        all_col = mydb['dictionary']
        all_col_raw = mydb['dictionary_raw']
        
        scrpr.all_col_toke = mydb['dictionary_token']
        topics_dict = { "title": submission.title , "score":submission.score, 
                       "id":submission.id, "url":submission.url, 
                       "comms_num":submission.num_comments , 
                       "body":submission.selftext, 
                       "z_comments":[]}    
        mycol.insert_one(topics_dict) 
        title = str(submission.title)
        '''title needs to be sliced due to character limitations'''
        title = title[0:20]
        new_col = mydb[title]
#Step 3: Scrape top "N" submission comments and store in mongoDB
        submission.comments.replace_more(limit=None)
        for comment in submission.comments.list():
            mydict1 = {"message":comment.body}
            raw_comment = {"message":comment.body}
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
            token_filtered_sent = {"message":filtered_sentence}
            scrpr.all_col_toke.insert_one(token_filtered_sent)
            filtered_sentence = ' '.join(filtered_sentence)
#Step 10: Store processed messages in mongoDB        
            mydict2 = {"message":filtered_sentence}
            all_col.insert_one(mydict2)
        print("Done", submission.title)
            #comment_body =  comment_body + comment.body + "\n"
    print("Done scraping")
    
    

def wrdfrq(all_col_toke):
    '''takes dataframe with tokenized messages as variable input,
    if scrape_reddit() was run, it can take variable all_col_toke'''
    #Step 0: Prepare libraries, database, and dataframes
    import pandas as pd
    from collections import defaultdict
    frequency = defaultdict(int)
    new_frequency = defaultdict(int)
    data_token = pd.DataFrame(list(all_col_toke.find()))
    processed_messages = []
    #Step 1: Count words and record frequency
    for messages in data_token['message']:
        for text in messages:
            frequency[text] += 1 
    #Step 2: Assuming words appearing once are unimportant, removed words appearing
    #less than 2 times
        processed_corpus = [text for text in messages if frequency[text] >2]
        for text in processed_corpus:
            new_frequency[text] += 1
    #Step 3: Take filtered messages and create final list
        processed_messages.append(processed_corpus)
    #Step 4: Convert final frequncy data into a dataframe   
    wordfrequency = pd.DataFrame([new_frequency])
    wordfrequency= wordfrequency.T
    wordfrequency= wordfrequency.reset_index()
    wrdfrq.wordfrequency = wordfrequency.rename(columns= {"index" : "word", 0 : "Count"})
    wrdfrq.processed_messages = processed_messages
    return processed_messages, len(new_frequency)
def tfidf(query, tokenized_list):
    #Step 1: Create corpus from processed messages(this can be accessed from 
    #wrdfrq.processed_messages)    
    import gensim as gm
    import pandas as pd
    dictionary = gm.corpora.Dictionary(tokenized_list)
    bow_corpus = [dictionary.doc2bow(text) for text in tokenized_list]
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
    print(result.head())    
#

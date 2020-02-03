#Initialise Database + Reddit
import os
from dotenv import load_dotenv, find_dotenv

#Step 1: find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()

#Step 2: load up the entries as environment variables
load_dotenv(dotenv_path, override=True)

processed_path =  os.getenv("processed_path")
raw_path =  os.getenv("raw_path")


import pandas as pd
#import pprint
data_raw = pd.read_csv(raw_path+"raw.csv")
result = pd.read_csv(raw_path+"query_results.csv")
data = pd.read_csv(processed_path+"processed.csv")
#vader needs source comments, not preprocessed, comments need to be in a single list
#positive sentiment : (compound score >= 0.05)
#neutral sentiment : (compound score > -0.05) and (compound score < 0.05)
#negative sentiment : (compound score <= -0.05)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()


#Function can be called on data-sets
def print_sentiment_scores(sentence):
    file = open(raw_path+"VADER_REPORT_SUMMARY_print_sentiment_scores.txt", "w+")
    
    sentiment_dict  = analyser.polarity_scores(sentence)
    #pprint.pprint("{:-<40} {}".format(sentence, str(snt)))
    print("Overall sentiment dictionary is : ", sentiment_dict)
    #print(result['msg'][msgs])
    print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative") 
    print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral") 
    print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive") 
    file.write("Overall sentiment dictionary is : "+ str(sentiment_dict)+ '\n'+
               "sentence was rated as "+ str(sentiment_dict['neg']*100)+ "% Negative"+'\n'+
               "sentence was rated as "+ str(sentiment_dict['neu']*100)+ "% Neutral"+'\n'+
               "sentence was rated as "+ str(sentiment_dict['pos']*100)+ "% Positive"+'\n'+
               "Sentence Overall Rated As "+'\n')
    print("Sentence Overall Rated As", end = " ") 
  
    # decide sentiment as positive, negative and neutral 
    if sentiment_dict['compound'] >= 0.05 : 
        print("Positive") 
        file.write("Positive")
    elif sentiment_dict['compound'] <= - 0.05 : 
        print("Negative") 
        file.write("Negative")
    else : 
        print("Neutral") 
        file.write("Neutral")
    print("")
    file.close()
#######################    

print("Overall consensus of query")
single_msg_list = result['msg'].values.tolist()
single_msg_list = ' '.join(single_msg_list)
print_sentiment_scores(single_msg_list)
print("-----")


#classify all msgs individually using VADER for returned query#
q_sentiment_label = []
q_message_full = []
q_compound_score_full = []
q_pos = []
q_neu = []
q_neg = []

for msgs in range(0,len(result)):
    sentiment_dict  = analyser.polarity_scores(result['msg'][msgs])
    if sentiment_dict['compound'] >= 0.05 : 
        q_sentiment_label.append("positive") 
        q_message_full.append(result['msg'][msgs])
        q_compound_score_full.append(sentiment_dict['compound'])
        q_pos.append(sentiment_dict['pos']*100)
        q_neu.append(sentiment_dict['neu']*100)
        q_neg.append(sentiment_dict['pos']*100)
        
    elif sentiment_dict['compound'] <= - 0.05 : 
        q_sentiment_label.append("negative") 
        q_message_full.append(result['msg'][msgs])
        q_compound_score_full.append(sentiment_dict['compound'])
        q_pos.append(sentiment_dict['pos']*100)
        q_neu.append(sentiment_dict['neu']*100)
        q_neg.append(sentiment_dict['pos']*100)
    else :  
        q_sentiment_label.append("neutral") 
        q_message_full.append(result['msg'][msgs])
        q_compound_score_full.append(sentiment_dict['compound'])
        q_pos.append(sentiment_dict['pos']*100)
        q_neu.append(sentiment_dict['neu']*100)
        q_neg.append(sentiment_dict['pos']*100)
    
query_SA = pd.concat([pd.Series(q_sentiment_label, name = "sentiment_label"),
                        pd.Series(q_message_full, name = "msg"),
                        pd.Series(q_compound_score_full, name = "compound score"), 
                        pd.Series(q_pos, name = "% positive"), 
                        pd.Series(q_neu, name = "%neutral"), 
                        pd.Series(q_neg, name = "% negative")], axis=1)
    
query_SA.to_csv(raw_path+"query_VADER_SA_result.csv")


##################################################################################
##############CREATES TRAINING DATA FOR MACHINE LEARNING##########################
##################################################################################
    
sentiment_label = []
message_full = []
compound_score_full = []
process_msg = []
pos = []
neu = []
neg = []
for full_text_words in range(0, len(data_raw)):
    sent_train  = analyser.polarity_scores(data_raw['message'][full_text_words])
    if sent_train['compound'] >= 0.05 : 
        sentiment_label.append("positive") 
        message_full.append(data_raw['message'][full_text_words])
        process_msg.append(data['message'][full_text_words])
        compound_score_full.append(sent_train['compound'])
        pos.append(sent_train['pos']*100)
        neu.append(sent_train['neu']*100)
        neg.append(sent_train['pos']*100)
    elif sent_train['compound'] <= - 0.05 : 
        sentiment_label.append("negative") 
        message_full.append(data_raw['message'][full_text_words])
        process_msg.append(data['message'][full_text_words])
        compound_score_full.append(sent_train['compound'])
        pos.append(sent_train['pos']*100)
        neu.append(sent_train['neu']*100)
        neg.append(sent_train['pos']*100)
    else : 
        sentiment_label.append("neutral")  
        message_full.append(data_raw['message'][full_text_words])
        process_msg.append(data['message'][full_text_words])
        compound_score_full.append(sent_train['compound'])
        pos.append(sent_train['pos']*100)
        neu.append(sent_train['neu']*100)
        neg.append(sent_train['pos']*100)
train_data = pd.concat([pd.Series(sentiment_label, name = "sentiment_label"),
                        pd.Series(process_msg, name = "process_msg"), 
                        pd.Series(message_full, name = "msg"),
                        pd.Series(compound_score_full, name = "compound score"), 
                        pd.Series(pos, name = "% positive"), 
                        pd.Series(neu, name = "%neutral"), 
                        pd.Series(neg, name = "% negative")], axis=1)
train_data.to_csv(raw_path+"VADER_SA_Training.csv")        
##################################################################################
import pandas as pd
from encoder import Model
#from utils import sst_binary




sentiment_model = Model()

data_token = pd.read_csv("C:/Users/matthew li yuen fong/Desktop/sentiment-analysis-master/data/raw/raw.csv")

samples = list(data_token['message']) 
subsample = [samples[1]]
#samples1 = "I want to transition from support to midlane. Any tips on the lane and good tutorials? Also, i dont know how to manager waves, so i need some help here"
#s = "I want to transition from support to midlane. Any tips on the lane and good tutorials? Also, i dont know how to manager waves, so i need some help here"
#samples1 = sst_binary()
sent = []
for sublist in data_token['message']:
    subsample = [sublist]
    text_features = sentiment_model.transform(subsample)
    sentiment_scores = text_features[:, 2388]
    sent.append(sentiment_scores)
result = pd.DataFrame(data= {"sentiment":sent,"message": data_token['message'][0:3847]})
    
#data_token['sentiment_scores'] = sentiment_scores

#data_token.to_csv('C:/Users/matthew li yuen fong/Desktop/sentiment-analysis-master/data/raw/openaiscore.csv')





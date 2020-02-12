#Initialise Database + Reddit
import os
from dotenv import load_dotenv, find_dotenv

#Step 1: find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()

#Step 2: load up the entries as environment variables
load_dotenv(dotenv_path, override=True)

processed_path =  os.getenv("processed_path")
raw_path =  os.getenv("raw_path")
testing = os.getenv("testing")
from gensim.models import Word2Vec
from gensim.test.utils import datapath

import pandas as pd
import pprint


data_token = pd.read_csv(processed_path + "processed+tokenized.csv")
data_token['message'] = data_token['message'].apply(eval)

Model= Word2Vec(data_token['message'], window=3, workers=10, iter=30, sg=1, negative = 15, alpha = 0.75, min_count=5)
w1 = "tsm"
pprint.pprint(Model.wv.most_similar (positive = w1))


accuracy = Model.wv.evaluate_word_analogies(datapath(testing+'questions-words.txt'))

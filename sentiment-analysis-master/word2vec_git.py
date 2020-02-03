from gensim.models import Word2Vec
import pandas as pd
from nltk.tokenize import word_tokenize
df = pd.read_csv()
df.dropna(inplace=True, axis = 0)
df.reset_index(inplace=True)
#df.Index.dropna(inplace=True)
print(df.head(5))
Bigger_list = []

for messages in range(0, len(df)):
    print(messages)
    word_tokens_message = word_tokenize(df['process_msg'][messages])
    Bigger_list.append(word_tokens_message)
#    for text in word_tokens_message:
#        frequency[text] += 1
#
Model= Word2Vec(Bigger_list,size=300, window=20, min_count=2, negative =10,  workers=10, iter=30)
w1 = "tsm"
print(Model.wv.most_similar (positive = w1))

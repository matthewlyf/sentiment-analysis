#Initialise Database + Reddit
import os
from dotenv import load_dotenv, find_dotenv

#Step 1: find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()

#Step 2: load up the entries as environment variables
load_dotenv(dotenv_path, override=True)

processed_path =  os.getenv("processed_path")
raw_path =  os.getenv("raw_path")


import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import TrigramCollocationFinder
import pandas as pd

data_token = pd.read_csv(processed_path+"processed+tokenized.csv")
data_token['message'] = data_token['message'].apply(eval)


flat_list = []
for sublist in data_token['message']:
    for item in sublist:
        flat_list.append(item)
######################finds top bigrams and trigrams###########################

bigrams = nltk.collocations.BigramAssocMeasures()
trigrams = nltk.collocations.TrigramAssocMeasures()

trigramfinder = TrigramCollocationFinder.from_words(flat_list)
bigramfinder =BigramCollocationFinder.from_words(flat_list)

bigram_freq = bigramfinder.ngram_fd.items()
bigramFreqTable = pd.DataFrame(list(bigram_freq), columns=['bigram','freq']).sort_values(by='freq', ascending=False)

trigram_freq = trigramfinder.ngram_fd.items()
trigramFreqTable = pd.DataFrame(list(trigram_freq), columns=['trigram','freq']).sort_values(by='freq', ascending=False)

#####Test####
#get english stopwords
from nltk.corpus import stopwords 

en_stopwords = set(stopwords.words('english')) 
#function to filter for ADJ/NN bigrams
def rightTypes(ngram):
    if '-pron-' in ngram or 't' in ngram:
        return False
    for word in ngram:
        if word in en_stopwords or word.isspace():
            return False
    acceptable_types = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    second_type = ('NN', 'NNS', 'NNP', 'NNPS')
    tags = nltk.pos_tag(ngram)
    if tags[0][1] in acceptable_types and tags[1][1] in second_type:
        return True
    else:
        return False
    
#filter bigrams
filtered_bi = bigramFreqTable[bigramFreqTable.bigram.map(lambda x: rightTypes(x))]
#function to filter for trigrams
def rightTypesTri(ngram):
    if '-pron-' in ngram or 't' in ngram:
        return False
    for word in ngram:
        if word in en_stopwords or word.isspace():
            return False
    first_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    third_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    tags = nltk.pos_tag(ngram)
    if tags[0][1] in first_type and tags[2][1] in third_type:
        return True
    else:
        return False
#filter trigrams
filtered_tri = trigramFreqTable[trigramFreqTable.trigram.map(lambda x: rightTypesTri(x))]
######


#filter for only those with more than 20 occurences
bigramfinder.apply_freq_filter(20)
trigramfinder.apply_freq_filter(20)


#We can see that PMI picks up bigrams and trigrams that consist of words that should co-occur together.
bigramPMITable = pd.DataFrame(list(bigramfinder.score_ngrams(bigrams.pmi)), columns=['bigram','PMI']).sort_values(by='PMI', ascending=False)
trigramPMITable = pd.DataFrame(list(trigramfinder.score_ngrams(trigrams.pmi)), columns=['trigram','PMI']).sort_values(by='PMI', ascending=False)

bigramPMITable.to_csv(raw_path+"bigramPMITable.csv")
trigramPMITable.to_csv(raw_path+"trigramPMITable.csv")

bigramChiTable = pd.DataFrame(list(bigramfinder.score_ngrams(bigrams.chi_sq)), columns=['bigram','chi-sq']).sort_values(by='chi-sq', ascending=False)
trigramChiTable = pd.DataFrame(list(trigramfinder.score_ngrams(trigrams.chi_sq)), columns=['trigram','chi-sq']).sort_values(by='chi-sq', ascending=False)

bigramChiTable.to_csv(raw_path+"bigramChiTable.csv")
trigramChiTable.to_csv(raw_path+"trigramChiTable.csv")
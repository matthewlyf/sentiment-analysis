Abstract | 

Insights gained from sentiment analysis (SA) have allowed businesses and users an opportunity to gain a deeper understanding of how communities perceive products, brands, topics, and/or services. Reddit is an American social news aggregation, web content rating, and discussion website and as of Jan 2020 according to Alexa Internet, it ranks as the fifth most visited website in the Canada and 18th in the world.  For users, one advantage to using reddit is that all topics are aggregated in a sub community (known as a subreddit) where individuals may post and comment on topics of specific interest to them.  By employing a generalized sentiment analysis script, we aim to mine the text to develop a generalised model that will analyse submissions and comments so that we can gain insights on what is currently trending in any subreddit in real-time.  Currently there are 2 popular approaches for analyzing text, the first utilizes a bag of words method and applies TF-IDF while the second employs a shallow 2-layer neural network that are trained to reconstruct the contexts of words known as words2vec.  For the scope of this project, our methodology is as follows; 1) Utilise Reddit’s api to scrape subreddit submission titles and comments 2) pre-process (stop words, symbols, apostrophes, lemmatize ) and tokenize text 3) upload the data to a MongoDB database 4) Apply TF-IDF, words2vec, and sentiment analysis models on the pre-processed text 5) Deploy streaming version to get live updates of subreddits.

Introduction 
With the dramatic shift and prevalence of social media in modern society, text analytics can be a powerful tool for companies to take strategic action based on valuable insights on common themes and trends found on the internet. To make text analytics the most efficient, organisations can use text analytics software, leveraging machine learning and natural language processing algorithms to find meaning in enormous amounts of text.  Reddit is an American social news aggregation, web content rating, and discussion website and as of Jan 2020 according to Alexa Internet, it ranks as the fifth most visited website in the Canada and 18th in the world.  For users, one advantage to using reddit is that all topics are aggregated in a sub community (known as a subreddit) where individuals may post and comment on topics of specific interest to them.  Therefore, there is a wealth of commentary, opinions, and knowledge that can be mined from the submissions and comments of reddit that can have the potential to provide valuable insight to any company and/or topic of interest.  For the scope of this project, our methodology is as follows; 1) Utilise Reddit’s api to scrape subreddit submission titles and comments 2) pre-process (stop words, symbols, apostrophes, lemmatize ) and tokenize text 3) upload the data to a MongoDB database 4) Apply TF-IDF, words2vec, and sentiment analysis models on the pre-processed text 5) Deploy streaming version to get live updates of subreddits.


References:
1) https://radimrehurek.com/gensim/auto_examples/core/run_core_concepts.html#sphx-glr-auto-examples-core-run-core-concepts-py
Explores concepts directly related to using genism, such as 

Document: some text.
Corpus: a collection of documents.
Vector: a mathematically convenient representation of a document.
Model: an algorithm for transforming vectors from one representation to another

2) D. Jurafsky and M. James H., “Vector Semantics and Embeddings,” in Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition, 3rd ed. Stanford University, UK: Online, 2019, pp. 94–122.

How to represent words

Term document matrix – table mapping frequency of words to documents.  Ie query is tokenized and each word frequency is mapped to its appearance in each document. Term-doc matrices used in information retrieval where  Two documents that are similar will tend to have similar words, and if two documents have similar words their column vectors will tend to be similar. Information retrieval (IR) is the task of ﬁnding the document d from the D information retrieval documents in some collection (n) that best matches a query q.
Collocation/co-occurance/bigrams/trigrams can be identified using word-word matrices where the frequency of words around a target word is calculated.
Cosine similarity calculates the angle between 2 vectors. IN the case for text, we can find the angle between 2 words found through the frequencies in a co-occurance word-word matrix and determine if the two words are more or less similar to another pair.
TF-IDF : giving weight to terms in a given vector
Term frequency is the first aspect that gives weight to words while the second is how frequent it appears in documents, such that words that are useful for describing documents from rest of collection are limited, while those that occur frequently aren’t helpful.
TF-IDF is good for weighting co-occurrence matrices for IR
Pointwise mutual information, calculates the probability of   2 events (x and y) are to occur compared to what we would expect if they were to appear independently. PMI can range from negative to positive infinity though negative pmi values are unreliable (unless corpora is massive) as it suggests words are co-occuring less often than we would expect by chance.
It is common practice to replace all negative pmi values with 0 which is known as positive pmi (ppmi)

Word2vec
Compared to TF-idf where we ask how often each word occurs near a target word, we ask a binary prediction which is how likely is a word to appear near the target word.
By using the corpus as the supervised training data for a binary classifier words near the target word in the training data acts as the gold answer to the question.
Word2vec is a binary classification and is trained using a logistic regression classifier.  Word2vec uses the algorithm known as skip-gram which treats target word and neighboring context word as positive examples.  It randomly samples other words to get negative samples.  Uses a logistic regression model to train classifier to distinguish the 2 cases and uses the regression weights as embeddings.

The classifier takes a test target word and its context window (ie number of words around it) and assigns a probability based on how similar that window is to the target word which is based ona  logistic function.
1) Corpus is a collection of document objects, the corpus in this project will include a collection of comments.
2) Tokenize the document objects
3) Keep words that appear more than once
4) Assign IDs to unique words in the corpus
5) Represent each document as a vector of features (known as a dense vector) and remove any elements within the vector that consist of 0.0, which would indicate a word has zero occurrences.  This new vector is known as a sparse vector or bag-of-words vector.  Dimension of the dense vector is equal to the number of unique words, the dimension of a bag-of-words vector is the number of unique words found in document (comment).
6)Convert entire corpus as a bag-of-words vector
7) Apply TRANSAFORMATION, the tf-idf model transforms vectors from the bag-of-words representation to a vector space where the frequency counts are weighted according to the relative rarity of each word in the corpus
8) We then index the transformed corpus and can then process our query by tokenizing and vectorizing it.  The model will then return which documents are most similar (similarity score) 


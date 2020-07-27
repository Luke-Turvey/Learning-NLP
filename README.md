# NLP-Algorithms

In order to learn the concepts of NLP, I have created this repository to put code I created during my learning process. 
I worked through the book "Natural Language Processing in Action" whilst learning so I have used lots of text data from the nlpia github especially when applying algorithms to small sets of data.
However, I have also used many of the corpus's supplied by the NLTK library for my own implementation of certain algorithms when a large corpus was needed.
The code is not as efficient as just using the models pre-built in genism and sklearn but this is because I wanted to program the core of the NLP algorithms myself in order to understand them better. I have used libraries such as sklearn for non-NLP specific data manipulation such as preprocessing.

Here is a list of the algorithms/models/pipelines I have included in this repository.

**Keyword Extraction**
- Textrank

**Sentiment Analysis**
- Naive Bayes for Movie Review Sentiment Analysis Classifaction(Good/Bad)

**TF-IDF**
- Created code to compute the TF-IDF vectors for the Brown Corpus without sklearn
- Used sklearn package to create TF-IDF vectors

**Topic Analysis**
- Used sklearn PCA to compute Topic Vectors for sms spam data
- Used Linear Discriminant Analysis (LDA) to classify text into spam/non-spam topics.
- Used LSA (sklearn SVD) to reduce dimensionality of BOW vector for logistic regression classifier for spam/non-spam sms.
- Created a plot of text reconstruction error against no. of dimensions eliminated by LSA
- Used LDiA to reduce dimensionality of BOW data and then LDA to classify text data as spam/not spam








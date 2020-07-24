from nltk.corpus import brown
from collections import Counter
import pandas as pd
import numpy as np

corpus = {}

categories = brown.categories()

for category in categories:
    tokens = brown.words(categories=str(category))
    doc_length = len(tokens)
    token_counts = dict(Counter(tokens))
    for token, value in token_counts.items():
        token_counts[token] = value/doc_length
    corpus[str(category)] = token_counts

tf = pd.DataFrame.from_records(corpus).fillna(0).T

idf = {}

num_docs = len(categories)

for index,word in enumerate(list(tf.columns)):
    num_docs_with_word = 0
    for category in categories:
        if tf.loc[category,word] != 0:
            num_docs_with_word+=1
    idf[str(word)] = num_docs/num_docs_with_word

inv_doc_freq = pd.DataFrame.from_dict({'idf': idf}).T

cols = tf.columns
inv_doc_freq=inv_doc_freq[cols]
inv_doc_freq=np.log10(inv_doc_freq)


tf_idf_matrix = pd.DataFrame(tf.values*inv_doc_freq.values, columns=tf.columns, index=tf.index).T
print(tf_idf_matrix)

from nltk.tokenize import TreebankWordTokenizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tokenizer = TreebankWordTokenizer()

# Documents for LSA to be applied too
docs = ["NYC is the Big Apple.",
        "NYC is known as the Big Apple.",
        "I love NYC!",
        "I wore a hat to the Big Apple party in NYC.",
        "Come to NYC. See the Big Apple.",
        "Manhattan is called the Big Apple."
        "Manhattan is a big city for a small cat.",
        "The lion, a big cat, is the king of the jungle.",
        "I love my pet cat.",
        "I love New York City (NYC).",
        "Your dog chased my cat."]

# Lexicon to construct BOW
lexicon = ["cat", "dog", "apple","lion", "nyc", "love"]

# Construcing the BOW
df = pd.DataFrame(index=lexicon)
for index, doc in enumerate(docs):
    doc_bow = []
    doc_tokens = tokenizer.tokenize(doc.lower())
    for vocab in lexicon:
        if vocab in doc_tokens:
            doc_bow.append(1)
        else:
            doc_bow.append(0)

    df.insert(column=index,value=doc_bow,loc=index)

# SVD on Bag of Words
U, s, Vt = np.linalg.svd(df)

S = np.zeros((len(U),len(Vt)))
pd.np.fill_diagonal(S,s)
S = pd.DataFrame(S).round(2)


# Error of document reconstruction based on number of dimensions lost
err = []

for numdim in range(len(s),0,-1):
    S.iloc[numdim-1,numdim-1] = 0

    reconstructed_df = U.dot(S).dot(Vt)
    err.append(np.sqrt(((reconstructed_df-df).values.flatten() ** 2).sum() / np.product(df.shape)))

plt.plot(list(range(0,len(s))),1-np.array(err))
plt.xlabel('Dimensions Reduced')
plt.ylabel('Reconstruction Accuracy')
plt.title("LSA Model Accuracy")
plt.show()
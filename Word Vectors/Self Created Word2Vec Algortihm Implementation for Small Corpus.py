from nltk.tokenize import word_tokenize
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import SGD

# The small corpus I am using
docs = ["NYC is the Big Apple. ",
        "NYC is known as the Big Apple. ",
        "I love NYC! ",
        "I wore a hat to the Big Apple party in NYC. ",
        "Come to NYC. See the Big Apple. ",
        "Manhattan is called the Big Apple. "
        "Manhattan is a big city for a small cat. ",
        "The lion, a big cat, is the king of the jungle. ",
        "I love my pet cat. ",
        "I love New York City (NYC). ",
        "Your dog chased my cat. "]

# Creating the corpus tokens and lexicon
doc_tokens = [word_tokenize(doc.lower()) for doc in docs]
lexicon = set(word_tokenize("  ".join(docs).lower()))

# This function returns a list of one hot vectors for the words surrounding the specified word
def skip_gram_vectors(word,doc_tokens):
    inputs = []
    for doc in doc_tokens:
        if word in doc:
            for x in [-3,-2,-1,1,2,3]:
                try:
                    one_hot_vector = dict((w, 0) for w in lexicon)
                    i = doc.index(word) + x
                    one_hot_vector[doc[i]] = 1
                    inputs.append(one_hot_vector)
                except:
                    pass

    return inputs

# This is where the inputs, and outputs to the neural network are created
word_outputs = []
word_inputs = []

for word in lexicon:
    one_word_hot_vector = dict((w, 0) for w in lexicon)
    one_word_hot_vector[word]=1

    for output in skip_gram_vectors(word,doc_tokens):
        word_outputs.append(list(output.values()))
        word_inputs.append(list(one_word_hot_vector.values()))

x_train = np.array(word_inputs)
y_train = np.array(word_outputs)

# Here I create the neural network to train my inputs and outputs
# n is the dimension of the word vectors
model = Sequential()
n=10
model.add(Dense(n,input_dim=x_train.shape[1]))
model.add(Activation('sigmoid'))
model.add(Dense(x_train.shape[1]))
model.add(Activation('softmax'))
model.summary()

# Here I train the neural networks on my inputs/outputs
sgd = SGD(lr=0.1)
model.compile(loss='mean_squared_error',optimizer=sgd,metrics=['accuracy'])
model.fit(x_train,y_train,epochs=200)

# Here I extract the parameters which define the word vectors
word_vecs = np.array(model.get_weights()[0])

# Here I ask through the terminal which word you would like to see in word vector form
word = input("Which word from the lexicon would you like the word-vector for: ")
one_word_hot_vector = dict((w, 0) for w in lexicon)
one_word_hot_vector[word.lower()] = 1
owhv = np.array(list(one_word_hot_vector.values())).T
print(np.matmul(owhv,word_vecs))


import pandas as pd
from nlpia.data.loaders import get_data
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize.casual import casual_tokenize
from sklearn.decomposition import LatentDirichletAllocation as LDiA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split

pd.options.display.width=120

sms = get_data('sms-spam')
index = [f'sms{i}{"!"*j}' for (i, j) in zip(range(len(sms)), sms.spam)]
sms.index = index

counter = CountVectorizer(tokenizer=casual_tokenize)
bow_docs = pd.DataFrame(counter.fit_transform(sms.text).toarray(), index=index)
column_nums, terms = zip(*sorted(zip(counter.vocabulary_.values(),counter.vocabulary_.keys())))
bow_docs.columns = terms

ldia = LDiA(n_components=16, learning_method='batch')
columns = [f'topic{i}' for i in range(ldia.n_components)]

ldia16_topic_vectors = ldia.fit_transform(bow_docs)
ldia16_topic_vectors = pd.DataFrame(ldia16_topic_vectors,index=index,columns=columns)

X_train, X_test, y_train, y_test = train_test_split(ldia16_topic_vectors,sms.spam,test_size=0.4,random_state=271828)
lda = LDA(n_components=1)
lda = lda.fit(X_train, y_train)
print(lda.score(X_test, y_test))
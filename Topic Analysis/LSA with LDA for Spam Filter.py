import pandas as pd
from nlpia.data.loaders import get_data
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split

pd.options.display.width=120

sms = get_data('sms-spam')
index = [f'sms{i}{"!"*j}' for (i, j) in zip(range(len(sms)), sms.spam)]
sms.index = index

tfidf = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf_docs = tfidf.fit_transform(raw_documents=sms.text).toarray()
tfidf_docs = pd.DataFrame(tfidf_docs)
tfidf_docs = tfidf_docs - tfidf_docs.mean()

svd = TruncatedSVD(n_components=16,n_iter=100)
svd_topic_vectors = svd.fit_transform(tfidf_docs.values)

columns = [f'topic{i}' for i in range(svd.n_components)]
svd_topic_vectors = pd.DataFrame(svd_topic_vectors,columns=columns,index=index)

X_train, X_test, y_train, y_test = train_test_split(svd_topic_vectors, sms.spam)
lda = LDA(n_components=1)
lda = lda.fit(X_train, y_train)
print(lda.score(X_test, y_test))
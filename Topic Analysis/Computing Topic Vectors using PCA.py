import pandas as pd
from nlpia.data.loaders import get_data
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize
from sklearn.decomposition import PCA

pd.options.display.width=120

sms = get_data('sms-spam')
index = [f'sms{i}{"!"*j}' for (i, j) in zip(range(len(sms)), sms.spam)]
sms.index = index

tfidf = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf_docs = tfidf.fit_transform(raw_documents=sms.text).toarray()
tfidf_docs = pd.DataFrame(tfidf_docs)
tfidf_docs = tfidf_docs - tfidf_docs.mean()

pca = PCA(n_components=16)
pca = pca.fit(tfidf_docs)
pca_topic_vectors = pca.transform(tfidf_docs)

columns = [f'topic{i}' for i in range(pca.n_components)]
pca_topic_vectors = pd.DataFrame(pca_topic_vectors,columns=columns,index=index)
print(pca_topic_vectors.round(3).head(10))

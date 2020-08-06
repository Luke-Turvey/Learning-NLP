import pandas as pd
from nlpia.data.loaders import get_data
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split

sms = get_data('sms-spam')
index = [f"sms{i}{'!'*j}" for (i, j) in zip(range(len(sms)), sms.spam)]

sms = pd.DataFrame(sms.values, columns=sms.columns, index=index)

sms['spam'] = sms.spam.astype(int)

tfidf_model = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf_docs = tfidf_model.fit_transform(raw_documents=sms.text).toarray()

X_train, X_test, y_train, y_test = train_test_split(tfidf_docs,sms.spam,test_size=0.33)

lda = LDA(n_components=1)
lda.fit(X_train,y_train)

print(lda.score(X_test,y_test))

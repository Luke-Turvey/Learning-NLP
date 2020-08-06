import pandas as pd
from nlpia.data.loaders import get_data
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

sms = get_data('sms-spam')
index = [f"sms{i}{'!'*j}" for (i, j) in zip(range(len(sms)), sms.spam)]

sms = pd.DataFrame(sms.values, columns=sms.columns, index=index)

sms['spam'] = sms.spam.astype(int)

tfidf_model = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf_docs = tfidf_model.fit_transform(raw_documents=sms.text).toarray()

X_train, X_test, y_train, y_test = train_test_split(tfidf_docs, sms.spam, test_size=0.4)

mask = y_train.astype(bool).values
spam_centroid = X_train[mask].mean(axis=0)
not_spam_centroid = X_train[~mask].mean(axis=0)

spam_line = spam_centroid - not_spam_centroid
spamminess_score = X_test.dot(spam_line)

LDA_Score = MinMaxScaler().fit_transform(spamminess_score.reshape(-1,1))
LDA_Predict = (LDA_Score > 0.5).astype(int)

classifier_score = (1.0 - (y_test - LDA_Predict.flatten()).abs().sum() / len(y_test))

print(f"Success Rate: {classifier_score}%")

import pandas as pd
from nlpia.data.loaders import get_data
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize
from sklearn.preprocessing import MinMaxScaler

sms = get_data('sms-spam')
index = [f"sms{i}{'!'*j}" for (i, j) in zip(range(len(sms)), sms.spam)]

sms = pd.DataFrame(sms.values, columns=sms.columns, index=index)

sms['spam'] = sms.spam.astype(int)

tfidf_model = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf_docs = tfidf_model.fit_transform(raw_documents=sms.text).toarray()

mask = sms.spam.astype(bool).values
spam_centroid = tfidf_docs[mask].mean(axis=0)
not_spam_centroid = tfidf_docs[~mask].mean(axis=0)

spam_line = spam_centroid - not_spam_centroid
spamminess_score = tfidf_docs.dot(spam_line)

sms['LDA_Score'] = MinMaxScaler().fit_transform(spamminess_score.reshape(-1,1))
sms['LDA_Predict'] = (sms.LDA_Score > 0.5).astype(int)

print(sms[['spam','LDA_Predict','LDA_Score']].head(10))

classifier_score = (1.0 - (sms.spam - sms.LDA_Predict).abs().sum() / len(sms))

print(f"Success Rate: {classifier_score}%")
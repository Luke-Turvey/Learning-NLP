from sklearn.feature_extraction.text import TfidfVectorizer
from nlpia.data.loaders import harry_docs as docs

vectoriser = TfidfVectorizer(min_df=1)
model=vectoriser.fit_transform(docs)
print(model.todense().round(2))

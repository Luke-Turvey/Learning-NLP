from nlpia.data.loaders import get_data
import pandas as pd
from nltk.tokenize import casual_tokenize
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


movies = get_data('hutto_movies')

print('Data Loaded...')

pd.set_option('display.width',75)
bag_of_words = []

for text in movies.text:
    bag_of_words.append(Counter(casual_tokenize(text)))

df_bows = pd.DataFrame.from_records(bag_of_words)
df_bows = df_bows.fillna(0).astype(int)

print('DataFrame created...')

X_train, X_test, y_train, y_test = train_test_split(df_bows, movies.sentiment > 0)

nb = MultinomialNB()
nb = nb.fit(X_train, y_train)


print(nb.score(X_test, y_test))
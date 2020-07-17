from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.cluster.util import cosine_distance
from nltk.corpus import stopwords
import numpy as np

text = """Turning once again, and this time more generally, to the question of invasion, I would observe that there has never been a period in all these long centuries of which we boast when an absolute guarantee against invasion, still less against serious raids, could have been given to our people. In the days of Napoleon, of which I was speaking just now, the same wind which would have carried his transports across the Channel might have driven away the blockading fleet. There was always the chance, and it is that chance which has excited and befooled the imaginations of many Continental tyrants. Many are the tales that are told. We are assured that novel methods will be adopted, and when we see the originality of malice, the ingenuity of aggression, which our enemy displays, we may certainly prepare ourselves for every kind of novel stratagem and every kind of brutal and treacherous manœuvre. I think that no idea is so outlandish that it should not be considered and viewed with a searching, but at the same time, I hope, with a steady eye. We must never forget the solid assurances of sea power and those which belong to air power if it can be locally exercised.

Sir, I have, myself, full confidence that if all do their duty, if nothing is neglected, and if the best arrangements are made, as they are being made, we shall prove ourselves once more able to defend our island home, to ride out the storm of war, and to outlive the menace of tyranny, if necessary for years, if necessary alone. At any rate, that is what we are going to try to do. That is the resolve of His Majesty's Government – every man of them. That is the will of Parliament and the nation. The British Empire and the French Republic, linked together in their cause and in their need, will defend to the death their native soil, aiding each other like good comrades to the utmost of their strength.


Even though large tracts of Europe and many old and famous States have fallen or may fall into the grip of the Gestapo and all the odious apparatus of Nazi rule, we shall not flag or fail. We shall go on to the end. We shall fight in France, we shall fight on the seas and oceans, we shall fight with growing confidence and growing strength in the air, we shall defend our island, whatever the cost may be. We shall fight on the beaches, we shall fight on the landing grounds, we shall fight in the fields and in the streets, we shall fight in the hills; we shall never surrender, and if, which I do not for a moment believe, this island or a large part of it were subjugated and starving, then our Empire beyond the seas, armed and guarded by the British Fleet, would carry on the struggle, until, in God's good time, the New World, with all its power and might, steps forth to the rescue and the liberation of the old."""


def text_clean(text):

    """This function takes in the text and cleans it by removing stopwords, short words
    and punctuation. It outputs an array of cleaned sentences and an array of cleaned words
    from the text."""

    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    stop_words = set(stopwords.words('english'))

    sentence_array = []

    word_array = [w.lower() for w in words if w.isalnum()]
    word_array = [w for w in word_array if w not in stop_words]
    word_array = [w for w in word_array if len(w) >= 2]

    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [w.lower() for w in words if w.isalpha()]
        words = [w for w in words if w not in stop_words]
        words = [w for w in words if len(w) >= 2]
        sentence_array.append(words)

    return sentence_array, word_array

def word_similiarity_matrix(text, k=4):

    """This function takes in the text for keyword extraction and creates a similiarity matrix
    for use in page-rank with chunks of size k in each sentence creating links between words."""

    sentence_array, word_array = text_clean(text)
    word_array = list(dict.fromkeys(word_array))
    wsm = np.zeros((len(word_array), len(word_array)), dtype=float)

    for sentence in sentence_array:
        if len(sentence) > k:
            for position in range(len(sentence)-k):
                chunk = sentence[position:position+k]
                positions = [word_array.index(word) for word in chunk]
                for i in positions:
                    for j in positions:
                        if i == j:
                            continue
                        wsm[i][j] = 1
        else:
            positions = [word_array.index(word) for word in sentence]
            for i in positions:
                for j in positions:
                    wsm[i][j] = 1

    wsm = wsm.T + wsm - np.diag(wsm.diagonal())

    totals = np.sum(wsm, axis=0)
    normalised_matrix = np.divide(wsm, totals, where=wsm!=0.0)

    return normalised_matrix

def word_rank(text):

    """This function takes in text and outputs a vector of ranks for each word in the text
    using the page-rank algorithm with the word_similiarity_matrix."""

    wsm = word_similiarity_matrix(text)
    sentence_array, word_array = text_clean(text)

    rank_vector = np.array([1] * len(set(word_array)))

    d=0.85
    iter = 50

    for _ in range(iter):
        rank_vector = 1 - d + d * np.matmul(wsm, rank_vector)

    return rank_vector

def key_words(text, n=10):

    """This function takes in text and uses the rank_vector from the word_rank function to
    output a list of n keywords from the text."""

    rank_vector = word_rank(text)
    np.round(rank_vector, 5, out=rank_vector)

    sentence_array, word_array = text_clean(text)
    word_array = list(dict.fromkeys(word_array))
    sorted_ranks = list(rank_vector)
    sorted_ranks.sort(reverse=True)

    key_word_list = []

    for i in range(n):
        if i < len(rank_vector):
            keyword = word_array[list(rank_vector).index(sorted_ranks[i])]
            key_word_list.append(keyword)
    return key_word_list


print(key_words(text,20))

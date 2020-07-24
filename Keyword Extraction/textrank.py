from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import numpy as np


class TextRank:

    def __init__(self,text):
        self.tokens = word_tokenize(text)
        self.sentences = sent_tokenize(text)

    def clean_text(self):
        stop_words = set(stopwords.words('english'))
        tokens = [w.lower() for w in self.tokens if w.isalnum() and w not in stop_words and len(w)>=2]

        sentences = []
        for sentence in self.sentences:
            words = [w.lower() for w in word_tokenize(sentence) if w.isalnum() and w not in stop_words and len(w)>=2]
            sentences.append(words)

        self.tokens = tokens
        self.sentences = sentences

    def word_similiarity_matrix(self,k=4):
        word_array = list(dict.fromkeys(self.tokens))
        wsm = np.zeros((len(word_array), len(word_array)), dtype=float)

        for sentence in self.sentences:
            if len(sentence) > k:
                for position in range(len(sentence) - k):
                    chunk = sentence[position:position + k]
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
        normalised_matrix = np.divide(wsm, totals, where=wsm != 0.0)

        return normalised_matrix

    def word_rank(self):

        wsm = self.word_similiarity_matrix()

        rank_vector = np.array([1] * len(set(self.tokens)))

        d = 0.85
        iteration = 50

        for _ in range(iteration):
            rank_vector = 1 - d + d * np.matmul(wsm, rank_vector)

        return rank_vector

    def keywords(self,n=25):
        rank_vector = self.word_rank()

        word_array = list(dict.fromkeys(self.tokens))
        sorted_ranks = list(rank_vector)
        sorted_ranks.sort(reverse=True)

        key_word_list = []

        for i in range(n):
            if i < len(rank_vector):
                keyword = word_array[list(rank_vector).index(sorted_ranks[i])]
                key_word_list.append(keyword)
        return key_word_list

text = """The Solanaceae, or nightshades, are a family of flowering plants that ranges from annual and perennial herbs to vines, lianas, epiphytes, shrubs, and trees, and includes a number of agricultural crops, medicinal plants, spices, weeds, and ornamentals. Many members of the family contain potent alkaloids, and some are highly toxic, but many—including tomatoes, potatoes, eggplant, bell and chili peppers—are used as food. The family belongs to the order Solanales, in the asterid group and class Magnoliopsida (dicotyledons).[2] The Solanaceae consists of about 98 genera and some 2,700 species,[3] with a great diversity of habitats, morphology and ecology.

The name Solanaceae derives from the genus Solanum, "the nightshade plant". The etymology of the Latin word is unclear. The name may come from a perceived resemblance of certain solanaceous flowers to the sun and its rays. At least one species of Solanum is known as the "sunberry". Alternatively, the name could originate from the Latin verb solare, meaning "to soothe", presumably referring to the soothing pharmacological properties of some of the psychoactive species of the family.

The family has a worldwide distribution, being present on all continents except Antarctica. The greatest diversity in species is found in South America and Central America. In 2017, scientists reported on their discovery and analysis of a fossil tomatillo found in the Patagonian region of Argentina, dated to 52 million years B.P. The finding has pushed back the earliest appearance of the plant family Solanaceae.[4] As tomatillos likely developed later than other nightshades, this may mean that the Solanaceae may have first developed during the Mesozoic Era.[5]

The Solanaceae include a number of commonly collected or cultivated species. The most economically important genus of the family[citation needed] is Solanum, which contains the potato (S. tuberosum, in fact, another common name of the family is the "potato family"), the tomato (S. lycopersicum), and the eggplant or aubergine (S. melongena). Another important genus, Capsicum, produces both chili peppers and bell peppers.

The genus Physalis produces the so-called groundcherries, as well as the tomatillo (Physalis philadelphica), the Cape gooseberry and the Chinese lantern. The genus Lycium contains the boxthorns and the wolfberry Lycium barbarum. Nicotiana contains, among other species, tobacco. Some other important members of Solanaceae include a number of ornamental plants such as Petunia, Browallia, and Lycianthes, and sources of psychoactive alkaloids, Datura, Mandragora (mandrake), and Atropa belladonna (deadly nightshade). Certain species are widely known for their medicinal uses, their psychotropic effects, or for being poisonous.

Most of the economically important genera are contained in the subfamily Solanoideae, with the exceptions of tobacco (Nicotiana tabacum, Nicotianoideae) and petunia (Petunia × hybrida, Petunioideae).

Many of the Solanaceae, such as tobacco and petunia, are used as model organisms in the investigation of fundamental biological questions at the cellular, molecular, and genetic levels."""

textrank = TextRank(text)
textrank.clean_text()
print(textrank.keywords())

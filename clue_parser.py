# Script takes in a list of crossword clues, and for each clue
# returns a list of important words from within the clue, as well as
# words returned from WordNetLemmatizer

import nltk, re, pprint
from nltk.corpus import stopwords, wordnet
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def tsv_to_list(clue_file):
    f = open(clue_file, 'r')
    lst = f.readlines()
    f.close()

    clu = list()
    length = list()
    ans = list()
    for i in range(len(lst)):
        temp = lst[i].split('\t')
        clu.append(temp[0])
        length.append(temp[1])
        ans.append(temp[2])

    return clu


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


file = open("keywords.txt", 'w')

for clue in tsv_to_list("allClues.tsv"):
    keywords = set([])
    clue = word_tokenize(clue.lower())
    filtered_clue = [w for w in clue if w not in stop_words]

    for word in filtered_clue:
        keywords.add(word)
        keywords.add(lemmatizer.lemmatize(word, get_wordnet_pos(word)))

    for word in keywords:
        file.write(word + " ")

    file.write("\n\n")

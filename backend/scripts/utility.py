#  Utility.py
# Purpose: File for utility functions that other files can use (i.e tagging words, calculating tf_idf, word normalization)
from nltk import sent_tokenize, word_tokenize, pos_tag, WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
import nltk
import math
CORPUS_SIZE = 37000

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

def get_wordnet_pos(treebank_tag):
    """Converts treebank POS tags to WordNet POS tags."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        # Default to noun if no match
        return wordnet.NOUN

def tag_words(list_of_words):
    list_of_words = word_tokenize(list_of_words)
    filtered_word_list = filter_words(list_of_words)
    normalized_word_list = normalize_word_list(filtered_word_list)
    tagged_words = pos_tag(normalized_word_list)
    return tagged_words

def calculate_tf_idf(numOfDocs, frequency):
    tf = 1 + math.log(frequency)
    idf = math.log(CORPUS_SIZE / numOfDocs ) #divided by postings list length
    tf_idf = tf * idf
    return tf_idf


                
def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        pass

def filter_words(list_of_words):
    for word in list_of_words:
        if word in stop_words:
            list_of_words.remove(word)
    return list_of_words
    #remove stop words

def normalize_word_list(list_of_words):
    for word in list_of_words:
        word = word.lower()
    return list_of_words

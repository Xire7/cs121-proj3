# Script for running through webpages and extracting the html content
import json
from nltk.corpus import wordnet, stopwords
import re
from bs4 import BeautifulSoup
import math
from nltk import sent_tokenize, word_tokenize, pos_tag, WordNetLemmatizer
import nltk
from nltk.stem import WordNetLemmatizer

web_directory = 'webpages/WEBPAGES_RAW/'
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


def run_and_extract():
    """
    Reads input from bookkeeping.json, locates each file, and attempts to parse each document
    """
    with open(web_directory+"bookkeeping.json", 'r') as file:
        data = json.load(file)
        for key in data: 
            # Getting the text so that it can be tokenized, lemmatized, and indexed
            with open(web_directory+key, 'r', encoding='utf-8') as file:
                content = file.read()
                soup = BeautifulSoup(content, 'html.parser')
                text = soup.get_text()
                #Passes the parsed HTML to create_index
                create_index(text, data[key]
                # list_of_sent = sent_tokenize(text) #list of sentences
                # lemmatizer = WordNetLemmatizer()
                # for sent in list_of_sent:
                #     list_of_words = word_tokenize(sent)
                #     filtered_word_list = filter_words(list_of_words)
                #     normalized_word_list = normalize_word_list(filtered_word_list)
                #     tagged_words = pos_tag(normalized_word_list)                    
                #     safe_print(tagged_words)
                #     length = len(tagged_words)
                #     for i in range(length):
                #         word_net_pos = get_wordnet_pos(tagged_words[i][1])
                #         lemmatized = lemmatizer.lemmatize(tagged_words[i][0], pos=word_net_pos)
                #         safe_print((tagged_words[i][0],lemmatized))
                    

def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        pass

def filter_words(list_of_words):
    for word in list_of_words:
        if word in stop_words:
            print("Found stop word:", word)
            list_of_words.remove(word)
    return list_of_words
    #remove stop words

def normalize_word_list(list_of_words):
    for word in list_of_words:
        word = word.lower()
    return list_of_words


def create_index(text, url):
    list_of_sent = sent_tokenize(text) #list of sentences
    lemmatizer = WordNetLemmatizer()
    for sent in list_of_sent:
        list_of_words = word_tokenize(sent)
        filtered_word_list = filter_words(list_of_words)
        normalized_word_list = normalize_word_list(filtered_word_list)
        tagged_words = pos_tag(normalized_word_list)
        safe_print(tagged_words)
        length = len(tagged_words)
        for i in range(length):
            word_net_pos = get_wordnet_pos(tagged_words[i][1])
            lemmatized = lemmatizer.lemmatize(tagged_words[i][0], pos=word_net_pos)
            safe_print((tagged_words[i][0],lemmatized))
            #add i(pos index) to DB
            if(i < length -1):
                two_gram = lemmatized + " " + lemmatizer.lemmatize(tagged_words[i+1][0], get_wordnet_pos(tagged_words[i+1][1]))
                store_in_db(two_gram, i)
            #store this into the DB along with i
            store_in_db(lemmatized, i)
        #pos of the word in sentence, store
        
    pass

def store_in_db(lemmatized, index):
    pass

#calculate analytics
def analytics():
    pass

def calculate_ranking(term, numOfDocs, termFrequency, docWordCount):
    # term = current word to calculate ranking of
    # numOfDocs (N) = Number of documents containing the term t
    # termFrequency (df) = frequency of term t in ENTIRE corpus
    # docWordCount (tf) = df / word count of ENTIRE corpus
    
    '''
    * tf-idf(t, d) = tf(t, d) * idf(t)
        N(t) = Number of documents containing the term t
        df(t) = occurrence of t in documents
    * tf(t,d) = count of t in d / number of words in d
    * idf(t) = log(N/ df(t))

    '''

    
    # numOfDocs (N) = Number of documents containing the term t
    # termFrequency (df) = frequency of term t in ENTIRE corpus
    # docWordCount (tf) = df / word count of ENTIRE corpus
    tf = termFrequency / docWordCount
    idf = math.log(numOfDocs / termFrequency)
    tf_idf = tf * idf

    print(f"TF-IDF of '{term}: {str(tf_idf)}'")
    
    #tf_idf = tf * idf

    pass


if __name__ == "__main__":
    run_and_extract()
    #create_index()
    #calculate_ranking()
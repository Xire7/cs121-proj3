# Script for running through webpages and extracting the html content
import json
import cbor2
import re
from cbor import load
from bs4 import BeautifulSoup
import math
from nltk import sent_tokenize, word_tokenize, pos_tag, WordNetLemmatizer
import nltk
from nltk.stem import WordNetLemmatizer

web_directory = 'webpages/WEBPAGES_RAW/'


"""
    Reads input from bookkeeping.json, locates each file, and attempts to parse each document
"""
def run_and_extract():
    # nltk.download('punkt')
    with open(web_directory+"bookkeeping.json", 'r') as file:
        data = json.load(file)
        for key in data: 
            # Getting the text so that it can be tokenized, lemmatized, and indexed
            with open(web_directory+key, 'r', encoding='utf-8') as file:
                # print("KEY:", key, "URL:", data[key])
                content = file.read()
                soup = BeautifulSoup(content, 'html.parser')
                text = soup.get_text()
                cleaned_text = clean_text(text)
                #Passes the parsed HTML to create_index

                list_of_sent = sent_tokenize(cleaned_text)
                safe_print(list_of_sent)
                # create_index(cleaned_text)
            
def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        pass


def clean_text(html_text):
    # Remove escape sequences (e.g. \n, \t, etc.) to make the word indexing more accurate
    html_text = re.sub(r'[\n\t\r\xa0]+', ' ', html_text)  # Use a regex pattern that matches one or more occurrences
    html_text = re.sub(r' +', ' ', html_text).strip()  # Remove multiple spaces and strip leading/trailing spaces
    #lowercase and not stop word
    return html_text

def filter_words(text):
    pass
    #remove stop words

def create_index(text):
    list_of_sent = sent_tokenize(text) #list of sentences
    lemmatizer = WordNetLemmatizer()
    for sent in list_of_sent:
        list_of_words = word_tokenize(sent)
        #print(list_of_words)
        list_of_words[i] = clean_text(list_of_words[i])
        tagged_words = pos_tag(list_of_words)
        print(tagged_words)
        length = len(tagged_words)
        for i in range(length):
            tagged_words[i] = filter_words(tagged_words[i]) #temp placeholder
            #if is valid(not stop word) and change to lowercase
            #add i(pos index) to DB
            lemmatized = lemmatizer.lemmatize(tagged_words[i])
            if(i < length -1):
                two_gram = lemmatized + " " + lemmatizer.lemmatize(tagged_words[i+1])
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
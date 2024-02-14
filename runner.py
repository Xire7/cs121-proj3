# Script for running through webpages and extracting the html content
import json
from nltk.corpus import wordnet, stopwords
import re
from bs4 import BeautifulSoup
import math
from nltk import sent_tokenize, word_tokenize, pos_tag, WordNetLemmatizer
import nltk
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import pymongo



class IndexData:
    def __init__(self, frequency, occurences, pos_in_sentence, pos_in_doc, ranking, url):
        self.occurences = occurences
        self.position_sentence = pos_in_sentence
        self.position_document = pos_in_doc
        self.ranking = ranking
        self.url = url


#class for storing analytics
        
class Analytics:
    def __init__(self):
        self.urls_discovered = set()
        self.url_count = 0
        self.unique_words = set()
        self.word_count = 0
        

    def update_word_count(self, word):
        if word not in self.unique_words:
            self.unique_words.add(word)
            self.word_count+=1
        
    def update_urls_discovered(self, url):
        if self.is_url_duplicate() is False:
            self.url_count+=1
            self.urls_discovered.add(url)
            
    def is_url_duplicate(self, url):
        return url in self.urls_discovered

            
result_analytics = Analytics()

#some form of viewable to_str method for displaying results
    

web_directory = 'webpages/WEBPAGES_RAW/'
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
inverted_index = defaultdict()
word_frequency_in_doc = defaultdict()
url_count = 0

# Initialize MongoDB client
client = pymongo.MongoClient("mongodb://localhost:27017/")
# Choose or create a database
db = client["search_engine"]
# Choose or create a collection
collection = db["inverted_index"]

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

    print(db.list_collection_names())
    if "inverted_index" in db.list_collection_names():
        safe_print("Collection exists, dropping again...")
        collection.drop()
        safe_print("Collection 'inverted_index' dropped.")
    
    with open(web_directory+"bookkeeping.json", 'r') as file:
        data = json.load(file)
        for key in data: 
            # Getting the text so that it can be tokenized, lemmatized, and indexed
            with open(web_directory+key, 'r', encoding='utf-8') as file:
                content = file.read()
                soup = BeautifulSoup(content, 'html.parser')

                special_words = extract_special_words(soup)
                text = soup.get_text()
                                
                #Passes the parsed HTML to create_index
                create_index(text, key, data[key], special_words)
                    

def extract_special_words(soup):
    special_words = {}
    relevance_tags = ['title', 'meta', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'b', 'strong', 'a'] #title gets 4 points, meta/h1 gets 3, h2-6/strong/b/a get 2, every other tag gets 1
    for tag in relevance_tags:
        for match in soup.find_all(tag): # find all the words that are in important tags, need later for positional index retrieval and ranking
            if tag in special_words:
                special_words[tag].append(match.get_text())
            else:
                special_words[tag] = [match.get_text()]
    return special_words
    
                
def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        pass

def filter_words(list_of_words):
    for word in list_of_words:
        if word in stop_words:
            #print("Found stop word:", word)
            list_of_words.remove(word)
    return list_of_words
    #remove stop words

def normalize_word_list(list_of_words):
    for word in list_of_words:
        word = word.lower()
    return list_of_words


def index_word(word, position, url, pos_in_doc, ranking,key): #add additional attr that indicates pos in doc
    if word not in inverted_index: #word hasn't appeared
        inverted_index[word] = [{key: [IndexData(frequency, url, position, pos_in_doc, ranking, url)]}]
    else:
        for dict in inverted_index[word]: # word has appeared but not this doc
            if url not in dict:
                inverted_index[word].append({url: [IndexData(frequency, url, position, pos_in_doc, ranking, key)]})
            else: #word appeared and in this doc
                inverted_index[word][url].append(IndexData(frequency, url, position, pos_in_doc, ranking, key))

def increment_frequency(word, docID):
    if word not in word_frequency_in_doc:
        word_frequency_in_doc[word] = {docID: 1}
    else:
        word_frequency_in_doc[word][docID] += 1
    
def create_index(text, key, url, special_words):
    if result_analytics.is_url_duplicate():
        return
    
    
    list_of_sent = sent_tokenize(text) #list of sentences

    lemmatizer = WordNetLemmatizer()

    word_position = 0
    for sent in list_of_sent:
        list_of_words = word_tokenize(sent)
        filtered_word_list = filter_words(list_of_words)
        normalized_word_list = normalize_word_list(filtered_word_list)
        tagged_words = pos_tag(normalized_word_list)
        # safe_print(f"tagged_words: (((( {tagged_words} ))))")
        ranking = 1
        length = len(tagged_words)
        for i in range(length): # range (i) is not accurate for entire doc position tracker because its only relative to start of sentences, use word_position instead
            word_net_pos = get_wordnet_pos(tagged_words[i][1])
            lemmatized = lemmatizer.lemmatize(tagged_words[i][0], pos=word_net_pos)
            #give ranking based off of lemmatized in special_words
            ranking = 2 # dummy val
            
            #safe_print((tagged_words[i][0],lemmatized))
            #add i(pos index) to DB
            if(i < length -1):
                two_gram = lemmatized + " " + lemmatizer.lemmatize(tagged_words[i+1][0], get_wordnet_pos(tagged_words[i+1][1]))
                index_word(frequency, two_gram, i, url, word_position, ranking, key)
                
            word_position += 1
            #store this into the DB along with i
            index_word(lemmatized, i, url, word_position, ranking, key)

            #Track word frequency 
            increment_frequency(lemmatized, key)
    
    # Store inverted index into MongoDB
    store_in_db(inverted_index)
            
        #pos of the word in sentence, store
        
        #need to return: a key for each term 



# Calling store_in_db
'''
    lemmatized: "lemmatized" from create_index
    docID: "key" from create_index
    frequency: word_frequency_in_doc[lemmatized][docID]
    occurences: what #th word in the document? 1st word? 10th word?
    pos_in_sentence: ???
    pos_in_doc: ???
    ranking: ???
    url: "url" from create_index

'''
def store_in_db(inverted_index):
    # Insert data into the MongoDB collection

    for term, info in inverted_index.items():
        
        lemmatized = term
        
        for nested_key, nested_value in info.items():
            
            docID = nested_key
            
            frequency = nested_value[0]
            occurences = nested_value[1]
            pos_in_sentence = nested_value[2]
            pos_in_doc = nested_value[3]
            ranking = nested_value[4]
            url = nested_value[5]
        
        
    safe_print(f'~ collection.insert_one( ([{lemmatized[0:5]}], [{docID}], [{url}]) )')
    
    # Check if the document already exists in the collection
    existing_doc = collection.find_one({'token': lemmatized})


    # If the document exists, update the list of URLs
    if existing_doc:
        doc_info = existing_doc['docs'].get(docID, {})
        doc_info['frequency'] = frequency
        doc_info['pos_in_sentence'] = pos_in_sentence
        doc_info['pos_in_doc'] = pos_in_doc
        doc_info['ranking'] = ranking
        doc_info['url'] = url
        existing_doc['docs'][docID] = doc_info
        # Update the document in the collection
        collection.update_one({'_id': existing_doc['_id']}, {'$set': {'docs': existing_doc['docs']}})
  
    # If the document does not exist, insert a new document
    else:
        collection.insert_one({
            'word': lemmatized,
            'docs': {
                docID: {
                    'frequency': frequency,
                    'occurences': occurences,
                    'pos_in_sentence': pos_in_sentence,
                    'pos_in_doc': pos_in_doc,
                    'ranking': ranking,
                    'url': url
                }
            }
        })

    safe_print(f'~ collection.insert_one DONE')

def get_from_db(phrase):
    # 1: lemmatize phrase into word?
    # search MongoDB for word?
    # pull the 10 H
w sLRU/sDIcoD deknaR tsehg
    pass
    
    

def display_db():
    # Query the database to retrieve all documents in the collection
    documents = collection.find()

    # Iterate over the documents and print them
    for document in documents:
        safe_print(f'DB Entry: "{document}"')

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
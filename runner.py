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
    def __init__(self, url, pos_in_sentence=0, pos_in_doc=0, ranking=0):
        self.url = url
        self.position_sentence = [pos_in_sentence]
        self.position_document = [pos_in_doc]
        self.ranking = ranking
        self.frequency = 1
        
    def add_pos_sentence(self, pos_sent):
        self.position_sentence.append(pos_sent)
    
    def add_pos_doc(self, pos_doc):
        self.position_document.append(pos_doc)

    def increment_frequency(self):
        self.frequency += 1

    def __str__(self):
        return (f"URL: {self.url}, "
                f"Positions in Sentence: {self.position_sentence}, "
                f"Positions in Document: {self.position_document}, "
                f"Ranking: {self.ranking}, "
                f"Frequency: {self.frequency}")
    
    def to_dict(self):
        return {
            'url': self.url,
            'position_sentence': self.position_sentence,
            'position_document': self.position_document,
            'ranking': self.ranking,
            'frequency': self.frequency
        }
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

        counter = 0
        for key in data: 
            # Getting the text so that it can be tokenized, lemmatized, and indexed
            with open(web_directory+key, 'r', encoding='utf-8') as file:
                content = file.read()
                soup = BeautifulSoup(content, 'html.parser')

                special_words = extract_special_words(soup)
                text = soup.get_text()
                                
                #Passes the parsed HTML to create_index
                create_index(text, key, data[key], special_words)

            ## For testing small document size ##
            counter += 1
            if counter == 10:
                print(f"Test size: {counter}")
                for term, docs in inverted_index.items():
                    safe_print(f"Term: {term}")
                    for doc_id, index_data in docs.items():
                        safe_print(f" Doc ID: {doc_id}, IndexData: {index_data}")    
                break
            ## Feel free to comment out ##
                    

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


def index_word(word, position, url, pos_in_doc, ranking, key):
    if word not in inverted_index: #word hasn't appeared
        inverted_index[word] = {key: IndexData(url, position, pos_in_doc, ranking)}
    else:
        if key not in inverted_index[word]: #changed from list of dictionaries to nested dictionary
            inverted_index[word][key] = IndexData(url, position, pos_in_doc, ranking)
        else:
            inverted_index[word][key].add_pos_sentence(position)
            inverted_index[word][key].add_pos_doc(pos_in_doc)
            inverted_index[word][key].increment_frequency()


        #every word has a list of dictionaries, in those dictionaries 
        # "poop": '0/0': ['https://poop.com', [1], [3,5], 1] , '0/1': ['https://roblox.com/awesomeness', [2], [4,10,20,92], 2] 


#another way if we keep postings as a list
# def index_word(word, position, url, pos_in_doc, ranking,key):
#     if word not in inverted_index: #word hasn't appeared
#         inverted_index[word] = [{key: [IndexData(url, position, pos_in_doc, ranking, url)]}]
#     else:
#         wordFound = False
#         for dict in inverted_index[word]: # word has appeared but not this doc
#             if url in dict:
#                 wordFound = True
#                 inverted_index[word][key].add_pos_sentence(position)
#                 inverted_index[word][key].add_pos_doc(pos_in_doc) #word appeared and in this doc
#                 inverted_index[word][key].increment_frequency()
#         if wordFound == False:
#             inverted_index[word].append({url: [IndexData(url, position, pos_in_doc, ranking)]})

    
def create_index(text, key, url, special_words):
    if result_analytics.is_url_duplicate(url):
        return
    
    list_of_sent = sent_tokenize(text) #list of sentences

    lemmatizer = WordNetLemmatizer()

    word_position = 0
    for sent in list_of_sent:
        list_of_words = word_tokenize(sent)
        filtered_word_list = filter_words(list_of_words)
        normalized_word_list = normalize_word_list(filtered_word_list)
        tagged_words = pos_tag(normalized_word_list)
        ranking = 1
        length = len(tagged_words)
        for i in range(length): # range (i) is not accurate for entire doc position tracker because its only relative to start of sentences, use word_position instead
            word_net_pos = get_wordnet_pos(tagged_words[i][1])
            lemmatized = lemmatizer.lemmatize(tagged_words[i][0], pos=word_net_pos)
            #give ranking based off of lemmatized in special_words
            ranking = 2 # dummy val
            
            if(i < length -1):
                two_gram = lemmatized + " " + lemmatizer.lemmatize(tagged_words[i+1][0], get_wordnet_pos(tagged_words[i+1][1]))
                index_word(two_gram, i, url, word_position, ranking, key)
                
            word_position += 1
            #store this into the DB along with i
            index_word(lemmatized, i, url, word_position, ranking, key)        
    


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
def store_in_db(documents):
    collection.insert_many(documents)

def prepare_documents_for_insertion(inverted_index):
    documents = []

    for token, postings in inverted_index.items():
        document = {
            'token': token,
            'postingsList': []
        }
        for doc_id, index_data in postings.items():
            posting = {
                'docId': doc_id, 
                'url': index_data.url,
                'sentencePosition': index_data.position_sentence,
                'documentPosition': index_data.position_document,
                'ranking': index_data.ranking,
                'frequency': index_data.frequency
                }
            document['postingsList'].append(posting)
        documents.append(document)
    return documents
    

def get_from_db(phrase):
    # 1: lemmatize phrase into word?
    # search MongoDB for word?
    # pull the 10 H
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
    documents = prepare_documents_for_insertion(inverted_index)
    store_in_db(documents)
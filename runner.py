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


web_directory = 'webpages/WEBPAGES_RAW/'
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
inverted_index = defaultdict()

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


# ok so basically each word has a postings list which is going to be a nested dictionary

def index_word(word, url, position):
    if word not in inverted_index:
        inverted_index[word] = {url: [position]}
    else:
        if url not in inverted_index[word]:
            inverted_index[word][url] = [position]
        else:
            inverted_index[word][url].append(position)
        


def create_index(text, key, url, special_words):
    
    list_of_sent = sent_tokenize(text) #list of sentences

    # for word in special_words:
    #     for match in special_words[word]:
    #         startingIndex = text.index(match) # TO-DO: captured all the words that are in special html tags, need to figure out how to get their positions in the text

    lemmatizer = WordNetLemmatizer()

    word_position = 0
    for sent in list_of_sent:
        list_of_words = word_tokenize(sent)
        filtered_word_list = filter_words(list_of_words)
        normalized_word_list = normalize_word_list(filtered_word_list)
        tagged_words = pos_tag(normalized_word_list)
        # safe_print(f"tagged_words: (((( {tagged_words} ))))")
        length = len(tagged_words)
        for i in range(length): # range (i) is not accurate for entire doc position tracker because its only relative to start of sentences, use word_position instead
            word_net_pos = get_wordnet_pos(tagged_words[i][1])
            lemmatized = lemmatizer.lemmatize(tagged_words[i][0], pos=word_net_pos)
            #safe_print((tagged_words[i][0],lemmatized))
            #add i(pos index) to DB
            if(i < length -1):
                two_gram = lemmatized + " " + lemmatizer.lemmatize(tagged_words[i+1][0], get_wordnet_pos(tagged_words[i+1][1]))
                # store_in_db(two_gram, word_position, url)
                index_word(two_gram, url, word_position)
            word_position += 1
            #store this into the DB along with i
            index_word(lemmatized, word_position, url)
            # store_in_db(lemmatized, word_position, url)

            
        #pos of the word in sentence, store
        
        #need to return: a key for each term 



def store_in_db(lemmatized, index, url):
    # Insert data into the MongoDB collection
    #data = {
    #    "token": lemmatized,
    #    "index": index,
    #    "url": url
    #}
    # invertedIndex = {
    #     "token": lemmatized,
    #     "postings": {
    #      }
    #}


    safe_print(f'~ collection.insert_one( ([{lemmatized[0:5]}], [{index}], [{url}]) )')
    #collection.insert_one(data)
    

    # Check if the document already exists in the collection
    existing_doc = collection.find_one({'token': lemmatized})


    # If the document exists, update the list of URLs
    if existing_doc:
        urls = existing_doc['urls']
        if url not in urls:
            urls.append(url)
        # Update the document in the collection
        collection.update_one({'_id': existing_doc['_id']}, {'$set': {'urls': urls}})
    # If the document does not exist, insert a new document
    else:
        collection.insert_one({

            'token': lemmatized,
            "index": index,
            'urls': [url]
        })

    safe_print(f'~ collection.insert_one DONE')

    

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


web_directory = 'webpages/WEBPAGES_RAW/'
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
inverted_index = defaultdict()

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
    relevance_tags = ['title', 'meta', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'b', 'strong', 'a'] #title gets 4 points, meta gets 3, h1 gets 3, h2-6/strong/b/a get 2, every other tag gets 1
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


# ok so basically each word has a postings list which is going to be a nested dictionary

def index_word(word, url, position):
    if word not in inverted_index:
        inverted_index[word] = {url: [position]}
    else:
        if url not in inverted_index[word]:
            inverted_index[word][url] = [position]
        else:
            inverted_index[word][url].append(position)
        


def create_index(text, key, url, special_words, source): #source is the documentID
    
    list_of_sent = sent_tokenize(text) #list of sentences

    # for word in special_words:
    #     for match in special_words[word]:
    #         startingIndex = text.index(match) # TO-DO: captured all the words that are in special html tags, need to figure out how to get their positions in the text

    lemmatizer = WordNetLemmatizer()
    for sent in list_of_sent:
        list_of_words = word_tokenize(sent)
        filtered_word_list = filter_words(list_of_words)
        normalized_word_list = normalize_word_list(filtered_word_list)
        tagged_words = pos_tag(normalized_word_list)
        # safe_print(f"tagged_words: (((( {tagged_words} ))))")
        length = len(tagged_words)
        ranking=1
        for i in range(length):
            word_net_pos = get_wordnet_pos(tagged_words[i][1])
            lemmatized = lemmatizer.lemmatize(tagged_words[i][0], pos=word_net_pos)
            #if is special word, deter ranking based on h1, h2, h3...
            #if lemmatized in special_words
            ranking = 2#dummy value
            
            #safe_print((tagged_words[i][0],lemmatized))
            #add i(pos index) to DB
            if(i < length -1):
                two_gram = lemmatized + " " + lemmatizer.lemmatize(tagged_words[i+1][0], get_wordnet_pos(tagged_words[i+1][1]))
                store_in_db(two_gram, i, url, source, ranking)
            #store this into the DB along with i
            store_in_db(lemmatized, i, url, source, ranking)

            
        #pos of the word in sentence, store
        
        #need to return: a key for each term 



def store_in_db(lemmatized, index, url, source, ranking=1): 
    # Insert data into the MongoDB collection
    data = {
        "word": lemmatized,
        "index": index,
        "url": url,
        "source": source,
        "ranking": ranking #default to 1 so when sorting, need to reverse sort
    }
    safe_print(f'~ collection.insert_one( ([{lemmatized[0:5]}], [{index}], [{url}]) )')
    collection.insert_one(data)
    safe_print(f'~ collection.insert_one DONE')
    

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

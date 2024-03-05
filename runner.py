# Script for running through webpages and extracting the html content
import json
from nltk.corpus import wordnet, stopwords
from bs4 import BeautifulSoup
import math
import string
from nltk import sent_tokenize, word_tokenize, pos_tag, WordNetLemmatizer
import nltk
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import pymongo
from pymongo.errors import OperationFailure
import ijson
import numpy as np
from urllib.parse import urljoin

CORPUS_SIZE = 37000

class IndexData:
    def __init__(self, url): # pos_in_sentence=0, pos_in_doc=0, ranking=0
        self.url = url
        self.tf_idf = 0
        self.tag_score = 0
        self.frequency = 1 
        # self.tag_list = [] # just for our visual reference
        
    def add_tag_score(self, tag):
        if tag == 'title':
            self.tag_score += 3
        elif tag == 'meta' or tag == 'h1' or tag == 'h2' or tag == 'h3':
            self.tag_score += 1.5
        elif tag == 'b' or tag == 'strong' or tag == 'a':
            self.tag_score += 1
        else:
            self.tag_score += 0.5
        # self.append_to_tag_list(tag)
        

    def increment_frequency(self):
        self.frequency += 1

    def __str__(self):
        return (f"URL: {self.url}, TAG_SCORE: {self.tag_score} FREQUENCY: {self.frequency}")
    
    def to_dict(self):
        return {
            'url': self.url,
        }
    
    def append_to_tag_list(self, tag):
        self.tag_list.append(tag)

    
    def get_tf_idf(self):
        print(f"TF-IDF: {str(self.tf_idf)}")
        return self.tf_idf
    
#class for storing analytics
        
class Analytics:
    def __init__(self):
        self.urls_discovered = set()
        self.url_count = 0
        self.unique_words = set()
        self.word_count = 0
        self.unique_docIds = set()
        

    def update_word_count(self, word):
        if word not in self.unique_words:
            self.unique_words.add(word)
            self.word_count+=1
        
    def update_urls_discovered(self, url):
        if self.is_url_duplicate(url) is False:
            self.url_count+=1
            self.urls_discovered.add(url)
            
    def is_url_duplicate(self, url):
        return url in self.urls_discovered
    
    def add_to_docIds(self, doc_id):
        self.unique_docIds.add(doc_id)

            
result_analytics = Analytics()
    

web_directory = 'webpages/WEBPAGES_RAW/'
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
inverted_index = defaultdict()
anchor_urls = defaultdict(list)


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

                special_words = extract_special_words(soup, data[key])
                text = soup.get_text()

                if data[key] in anchor_urls:
                    text += " ".join(anchor_urls[data[key]])
                                
                #Passes the parsed HTML to create_index
                create_index(text, key, data[key], special_words)

           ## For testing small document size ##
            # counter += 1
            # if counter == 100:
            #     # print(f"Test size: {counter}")
            #     # for term, docs in inverted_index.items():
            #     #     safe_print(f"Term: {term}")
            #     #     for doc_id, index_data in docs.items():
            #     #         safe_print(f" Doc ID: {doc_id}, IndexData: {index_data}")
            #     break
            # Feel free to comment out ##
                    


def extract_special_words(soup, url):
    special_words = {}
    relevance_tags = ['title', 'meta', 'h1', 'h2', 'h3', 'b', 'strong', 'a']
    for tag in relevance_tags:
        for match in soup.find_all(tag): # find all the words that are in important tags, need later for positional index retrieval and ranking
            matched_text = match.get_text()
            if matched_text.strip() == "": #skip empty tags
                continue
            if tag == 'a':
                if match.get('href') is not None:
                    anchor_urls[urljoin(url, match.get('href'))].append(match.get_text())
            if tag in special_words:
                special_words[tag].append(match.get_text())
            else:
                special_words[tag] = [match.get_text()]
            match.decompose() # removes tag from tree
    return special_words
    
                
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


def index_word(word, url, key, tag):
    result_analytics.update_word_count(word)
    result_analytics.add_to_docIds(key)
    if word not in inverted_index: #word hasn't appeared
        inverted_index[word] = {key: IndexData(url)} #{key: IndexData(url, position, pos_in_doc, ranking)} removing attributes to save space for now
    else:
        if key not in inverted_index[word]: #changed from list of dictionaries to nested dictionary
            inverted_index[word][key] = IndexData(url) # IndexData(url, position, pos_in_doc, ranking)
        else:
            inverted_index[word][key].increment_frequency()
    inverted_index[word][key].add_tag_score(tag)
    

def tag_words(list_of_words):
    list_of_words = word_tokenize(list_of_words)
    filtered_word_list = filter_words(list_of_words)
    normalized_word_list = normalize_word_list(filtered_word_list)
    tagged_words = pos_tag(normalized_word_list)
    return tagged_words

def create_index(text, key, url, special_words):
    if result_analytics.is_url_duplicate(url):
        return
    result_analytics.update_urls_discovered(url)
    
    list_of_sent = sent_tokenize(text) #list of sentences

    lemmatizer = WordNetLemmatizer()

    for sent in list_of_sent:
        tagged_words = tag_words(sent)
        length = len(tagged_words)
        for i in range(length): # range (i) is not accurate for entire doc position tracker because its only relative to start of sentences, use word_position instead
            word_net_pos = get_wordnet_pos(tagged_words[i][1])
            lemmatized = lemmatizer.lemmatize(tagged_words[i][0], pos=word_net_pos)            
            if(i < length -1):
                two_gram = lemmatized + " " + lemmatizer.lemmatize(tagged_words[i+1][0], get_wordnet_pos(tagged_words[i+1][1]))
                index_word(two_gram, url, key, "p")
                
            #store this into the DB along with i
            index_word(lemmatized, url, key, "p")      

    for tag, sent in special_words.items():
        for words in sent:
            tagged_words = tag_words(words)
            length = len(tagged_words)
            for i in range(length):
                word_net_pos = get_wordnet_pos(tagged_words[i][1])
                lemmatized = lemmatizer.lemmatize(tagged_words[i][0], pos=word_net_pos)
                if(i < length -1):
                    two_gram = lemmatized + " " + lemmatizer.lemmatize(tagged_words[i+1][0], get_wordnet_pos(tagged_words[i+1][1]))
                    index_word(two_gram, url, key, tag)
                index_word(lemmatized, url, key, tag)


# def store_in_db(documents): # storing inverted index into json file in case we lose information from the DB
#     with open('error_log.txt', 'a') as file:
#         with open('inverted_index.json', 'r') as inverted_index_file:
#             data = json.load(inverted_index_file)
#             for document in data:
#                 try:
#                     collection.insert_one(document)
#                 except Exception as e:
#                     file.write(f"{document['token']} too big to store: {len(document['postingsList']), {e}} \n")

            

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
                'tagScore': index_data.tag_score,
                'frequency': index_data.frequency
                }
            document['postingsList'].append(posting)
        documents.append(document)


    # with open('inverted_index.json', 'w') as file: #for storing in JSON file
    #     json.dump(documents, file)


    collection.insert_many(documents)

    return documents
    


def get_from_db(phrase_vector, query_vector = None):
    # search MongoDB for word?
    # pull the 20 entries with the highest Ranking

    # Search MongoDB for the word
    '''result = collection.find_one({'token': phrase})

    if result:
        # Sort postingsList by frequency in descending order
        sorted_postings = sorted(result['postingsList'], key=lambda x: x['frequency'], reverse=True)
        # Return the top 20 entries with the highest ranking
        return (sorted_postings[:20], len(sorted_postings))
    else:
        return None'''
    #get from db all docs that match each word -> store in data structure, use that data structure
    #later calc cosine similarity -> retrieve based off of top 20  formula: 80% cosine similarity + 10% tag + 10% pagerank cosine + tag
    

    #result = collection.find_one({'token': phrase})   # OLD

    result = []
    for phrase in phrase_vector:
        segment = collection.find_one({'token': phrase})    # find matching queries for EACH word in input phrase
        result.append(segment)                              # combine matching queries

    # query_vector = get_query()
    if result:
        postings = result['postingsList']
        ranked_postings = []
        for posting in postings:
            document_vector = np.array(list(posting['tf_idf']))  # Convert TF-IDF dictionary to array
            similarity = use_cosine_similarity(query_vector, document_vector)
            ranked_postings.append((posting, similarity))
        
        # Sort postings by cosine similarity in descending order
        ranked_postings.sort(key=lambda x: x[1], reverse=True)
        
        # Return the top 20 entries with the highest cosine similarity
        return (ranked_postings[:20], len(ranked_postings))
    else:
        return None
    
def get_top_entries():
    # Ask the user for input on the word

    query_result_list = []

    # phrase = input("Enter a phrase to search for: ")
    # Call get_from_db with the input word
    top_entries = get_from_db(phrase)
    query_result_list.append((phrase, top_entries))

    return query_result_list
    
def output_analysis(query_result_list):
    with open('analysis.txt','w', encoding='utf-8') as file:
        for query in query_result_list:
            if query[1] is not None:
                file.write(f"Total number of links for phrase {query[0]}: {query[1][1]} \n")
                file.write(f"Top 20 entries for phrase '{query[0]}':\n")
                for i, entry in enumerate(query[1][0], 1):
                    # file.write(f"{i}. Doc ID: {entry['docId']}, URL: {entry['url']}, Sentence Position: {entry['sentencePosition']}, Document Position: {entry['documentPosition']}, Ranking: {entry['ranking']}, Frequency: {entry['frequency']}\n")
                    file.write(f"{i}. Doc ID: {entry['docId']}, URL: {entry['url']} Frequency: {entry['frequency']}\n")

            else:
                file.write(f"No entries found for word '{query[0]}'.\n")


        file.write('\nAdditional analytics: \n')
        file.write(f"Number of Unique Words: {result_analytics.word_count}\n")
        file.write(f'URL Count = {result_analytics.url_count}\n')
        file.write(f'Number of unique documents = {len(result_analytics.unique_docIds)} \n')
        collection_stats = db.command("collStats", "inverted_index")

        # Extract the size from the stats
        collection_size_in_bytes = collection_stats['size']
        file.write(f'Database Size = {collection_size_in_bytes} KB')


def calculate_idf_from_mongo():
    documents = collection.find()
    for document in documents:
        postings = document['postingsList'] # postings is the List of Objects
        for posting in postings:
            print("POSTING:", posting, "\n")
            posting_value = calculate_tf_idf(len(postings), posting['frequency'])
            posting['tf_idf'] = posting_value
        collection.update_one({'_id': document['_id']}, {'$set': {'postingsList': postings}})
        print("Updated document", document['token'])
    return


def restore_db_from_json():
    with open('inverted_index.json', 'r') as file:
        counter = 0
        streamed_data = ijson.items(file, 'item')
        for item in streamed_data:
            posting_list = item['postingsList']
            print(posting_list)
            print(item['token'])
            collection.update_one({'token': item['token']}, {'$set': {'postingsList': posting_list} })
            counter += 1
            if counter == 1000: # at document 161, array size was 28
                break
        print("Data restored from JSON file.")
    

def display_db():
    # Query the database to retrieve all documents in the collection
    documents = collection.find()

    # Iterate over the documents and print them
    for document in documents:
        safe_print(f'DB Entry: "{document}"')


def calculate_tf_idf(numOfDocs, frequency):
    tf = 1 + math.log(frequency)
    idf = math.log(CORPUS_SIZE / numOfDocs ) #divided by postings list length
    tf_idf = tf * idf
    return tf_idf

def scoring(query):
    #float Scores[N] = 0
    #float Length[N] = 0
    scores = defaultdict(float)
    magnitude = defaultdict(float)
    cosine_similarity_scores = []
    
    for query_term in query:    #"query" being the words in the input.
        token = collection.find_one({'token': query_term})
        query_tf_idf = calculate_tf_idf(len(token['postingsList'], query.count(query_term)))
        for posting in token['postingsList']:
            scores[posting['docId']] += posting['tf_idf'] * query_tf_idf
            magnitude[posting['docId']] += posting['tf_idf'] ** 2

    #normalize vectors
    for docID, score in scores.items():
        cosine_score = score / math.sqrt(magnitude[docID])
        cosine_similarity_scores.append((docID, cosine_score))
    cosine_similarity_scores.sort(key=lambda x: x[1], reverse=True)
    return cosine_similarity_scores



    #Line 7: Update content of Length[N]
    for d in documents:                     # Line 8
        scores[d] = scores[d]/Length[d]     # Line 9
    #Line 10: return Top K components of Scores[]

def cosine_similarity(query_vector, document_vector):

    # if q and d are tf-idf vectors, then
    # cos = (q * d) / (||q|| * ||d||)

    """
    Computes the cosine similarity between two vectors.
    """
    dot_product = np.dot(query_vector, document_vector)
    query_norm = np.linalg.norm(query_vector)
    document_norm = np.linalg.norm(document_vector)
    similarity = dot_product / (query_norm * document_norm)
    return similarity
'''def compute_cosine_similarity(query_dict, document_vector):
    query_vector = np.array(list(query_dict.values()))  # Convert query_dict values to array
    return cosine_similarity(query_vector, document_vector)'''

# example usage
def use_cosine_similarity(document_vector):
    query_map = get_query()
    
    
    #get_from_db(query_map.keys())
    
    
    
    similarity = cosine_similarity(query_map.values(), document_vector)
    #update the db with cosine similarity

    #retreives top 20

    print("Cosine Similarity:", similarity)
    return query_vector

#query_vector: vector of tf-idfs 
#
# def make_query_length_same(query_vector, document_vector)


def calculate_proximity_weight(tokens, query_terms):
    """
    Calculate proximity weight for query terms in the document.
    Here, we use a simple approach: reciprocal of the positional difference between occurrences.
    """
    proximity_weights = {}
    for term in query_terms:
        positions = [i for i, token in enumerate(tokens) if token == term]
        if positions:
            proximity_weights[term] = sum(1 / (pos + 1) for pos in positions)
    return proximity_weights

def preprocess(text):
    """
    Tokenizes the text and removes stop words and punctuation.
    """
    # Tokenize the text
    tokens = text.lower().split()
    # Remove punctuation
    tokens = [token.strip(string.punctuation) for token in tokens]
    # Remove stop words
    tokens = [token for token in tokens if token not in stop_words]
    return tokens
def proximity_weighted_score(document, query_terms, document_tf_idf):
    """
    Calculate proximity-weighted score for the document.
    """
    tokens = preprocess(document)  # Preprocess the document (tokenization, removal of stop words, etc.)
    
    # Calculate proximity weights for query terms
    proximity_weights = calculate_proximity_weight(tokens, query_terms)
    
    # Compute proximity-weighted score
    score = 0
    for term, tf_idf in document_tf_idf.items():
        if term in proximity_weights:
            score += tf_idf * proximity_weights[term]
    
    return score
def get_proximity_score(documents, query_terms, document_tf_idf):
    for document in documents:  # Iterate over documents
        score = proximity_weighted_score(document, query_terms, document_tf_idf)
        print("Document:", document)
        print("Proximity-Weighted Score:", score)



def get_query():
    query = input("Enter what we need to search for: ")
    #sanitize query
    query = tag_words(query.split())
    #get tf-idf of all words in query
    lemmatizer = WordNetLemmatizer()
    query_len = len(query)
    frequency = {}
    query_terms = []
    for i in range(query_len - 1):
        lemmatized_word = lemmatizer.lemmatize(query[i][0], get_wordnet_pos(query[i][1]))
        frequency[lemmatized_word] += 1
        query_terms.append(lemmatized_word)
        second_word = lemmatizer.lemmative(query[i+1][0], get_wordnet_pos(query[i+1][1]))
        two_gram = lemmatized_word + " " + second_word
        frequency[two_gram] += 1
    lemmatized_word = lemmatizer.lemmatize(query[query_len - 1][0], get_wordnet_pos(query[query_len - 1][1]))
    frequency[lemmatized_word] += 1

    query_list={}


    scoring(query_terms)

    for key, value in frequency.values():
        result = collection.find_one({'token': key})
    
        if result:
            postings = result['postingsList']
            tf_idf = calculate_tf_idf(len(postings), value)
            query_list[key] = tf_idf

    return query_list  #returns a list

    # 

    # retrieve based off of top 20  formula: 80% cosine similarity + 10% tag + 10% pagerank


    #calculate cosine similarity

if __name__ == "__main__":
    # get_query()
    # run_and_extract()
    # documents = prepare_documents_for_insertion(inverted_index)
    calculate_idf_from_mongo()
    # add_special_words_to_index()

    #getting query and outputing
    
    # user_entries = get_top_entries()
    # output_analysis(user_entries)
    # store_in_db(documents) # probably not gonna need to store in json unless we need to store actual word positions
    # restore_db_from_json()
<<<<<<< HEAD
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

CORPUS_SIZE = 37000

class IndexData:
    def __init__(self, url): # pos_in_sentence=0, pos_in_doc=0, ranking=0
        self.url = url
        # self.position_sentence = [pos_in_sentence]
        # self.position_document = [pos_in_doc]
        #self.ranking = ranking
        self.tf_idf = 0
        self.frequency = 1  #in document
        
    # def add_pos_sentence(self, pos_sent):
    #     self.position_sentence.append(pos_sent)
    
    # def add_pos_doc(self, pos_doc):
    #     self.position_document.append(pos_doc)

    def increment_frequency(self):
        self.frequency += 1

    def __str__(self):
        return (f"URL: {self.url}, ")
                # f"Positions in Sentence: {self.position_sentence}, "
                # f"Positions in Document: {self.position_document}, "
                # f"Ranking: {self.ranking}, "
                # f"Frequency: {self.frequency}")
    
    def to_dict(self):
        return {
            'url': self.url,
        }
        #     'position_sentence': self.position_sentence,
        #     'position_document': self.position_document,
        #     'ranking': self.ranking,
        #     'frequency': self.frequency
        # }
    

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


# Initialize MongoDB client
client = pymongo.MongoClient("mongodb://localhost:27017/")
# Choose or create a database
db = client["search_engine"]
# Choose or create a collection
collection = db["inverted_index_test"] #changed to test

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

    if "inverted_index_test" in db.list_collection_names():
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

            # For testing small document size ##
            # counter += 1
            # if counter == 1000:
            #     print(f"Test size: {counter}")
                # for term, docs in inverted_index.items():
                #     safe_print(f"Term: {term}")
                #     for doc_id, index_data in docs.items():
                #         safe_print(f" Doc ID: {doc_id}, IndexData: {index_data}")    
                # break
            ## Feel free to comment out ##
                    


def extract_special_words(soup):
    special_words = {}
    relevance_tags = ['title', 'meta', 'h1', 'h2', 'h3', 'b', 'strong', 'a'] #title gets 4 points, meta/h1/h2/h3 gets 3, strong/b/a get 2, every other tag gets 1
    for tag in relevance_tags:
        for match in soup.find_all(tag): # find all the words that are in important tags, need later for positional index retrieval and ranking
            if match.get_text() == "": #skip empty tags
                continue
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
            list_of_words.remove(word)
    return list_of_words
    #remove stop words

def normalize_word_list(list_of_words):
    for word in list_of_words:
        word = word.lower()
    return list_of_words


def index_word(word, position, url, pos_in_doc, ranking, key):
    result_analytics.update_word_count(word)
    result_analytics.add_to_docIds(key)
    if word not in inverted_index: #word hasn't appeared
        inverted_index[word] = {key: IndexData(url)} #{key: IndexData(url, position, pos_in_doc, ranking)} removing attributes to save space for now
    else:
        if key not in inverted_index[word]: #changed from list of dictionaries to nested dictionary
            inverted_index[word][key] = IndexData(url) # IndexData(url, position, pos_in_doc, ranking)
        else:
            # inverted_index[word][key].add_pos_sentence(position)
            # inverted_index[word][key].add_pos_doc(pos_in_doc)
            inverted_index[word][key].increment_frequency()
    

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

    word_position = 0
    for sent in list_of_sent:
        tagged_words = tag_words(sent)
        ranking = 1
        length = len(tagged_words)
        for i in range(length):
            word_net_pos = get_wordnet_pos(tagged_words[i][1])
            lemmatized = lemmatizer.lemmatize(tagged_words[i][0], pos=word_net_pos)            
            if(i < length -1):
                two_gram = lemmatized + " " + lemmatizer.lemmatize(tagged_words[i+1][0], get_wordnet_pos(tagged_words[i+1][1]))
                index_word(two_gram, i, url, word_position, ranking, key)
                
            word_position += 1
            #store this into the DB along with i
            index_word(lemmatized, i, url, word_position, ranking, key)      

    for tag, sent in special_words.items():
        for words in sent:
            tagged_words = tag_words(words)
            length = len(tagged_words)
            for i in range(length):
                word_net_pos = get_wordnet_pos(tagged_words[i][1])
                lemmatized = lemmatizer.lemmatize(tagged_words[i][0], pos=word_net_pos)
                if(i < length -1):
                    two_gram = lemmatized + " " + lemmatizer.lemmatize(tagged_words[i+1][0], get_wordnet_pos(tagged_words[i+1][1]))
                    index_word(two_gram, i, url, word_position, ranking, key)
                word_position += 1
                index_word(lemmatized, i, url, word_position, ranking, key)


def store_in_db(documents): # storing inverted index into json file in case we lose information from the DB
    with open('error_log.txt', 'a') as file:
        with open('inverted_index.json', 'r') as inverted_index_file:
            data = json.load(inverted_index_file)
            for document in data:
                try:
                    collection.insert_one(document)
                except Exception as e:
                    file.write(f"{document['token']} too big to store: {len(document['postingsList']), {e}} \n")

def add_special_words_to_index(): # special words per document
    with open(web_directory+"bookkeeping.json", 'r') as file:
        data = json.load(file)

        counter = 0
        for key in data: 
            # Getting the text so that it can be tokenized, lemmatized, and indexed
            weight_dict = defaultdict(int)
            special_words_frequency = defaultdict(int)
            with open(web_directory+key, 'r', encoding='utf-8') as file:
                content = file.read()
                soup = BeautifulSoup(content, 'html.parser')
                special_words = extract_special_words(soup)

                # collection.update_one({"token": "5 3", "postingsList.docId": "0/0"}, {'$set': {"postingsList.$.ranking": 5}}) # this is the sample update

                # # Idea is to tag special words
                lemmatizer = WordNetLemmatizer()

                if not special_words:
                    print(special_words)
                    print("GG")
                    continue 

                for tag, sent in special_words.items():
                    for words in sent:
                        print("Sentence:", words, "Tag:", tag)
                        special_tokens = word_tokenize(words)
                        filtered_tokens = filter_words(special_tokens)
                        normalized_tokens = normalize_word_list(filtered_tokens)
                        tagged_words = pos_tag(normalized_tokens)
                        length = len(tagged_words)
                        for i in range(length):

                            word_net_pos = get_wordnet_pos(tagged_words[i][1])
                            lemmatized = lemmatizer.lemmatize(tagged_words[i][0], pos=word_net_pos)

                            if (tag == 'title'):
                                word_weight = 4
                            elif (tag == 'meta' or tag == 'h1' or tag == 'h2' or tag == 'h3'):
                                word_weight = 3
                            elif (tag == 'b' or tag == 'strong' or tag == 'a'):
                                word_weight = 2
                            if(i < length -1):
                                two_gram = lemmatized + " " + lemmatizer.lemmatize(tagged_words[i+1][0], get_wordnet_pos(tagged_words[i+1][1]))
                                weight_dict[two_gram] += word_weight
                                special_words_frequency[two_gram] += 1

                            weight_dict[lemmatized] += word_weight # aggregating weight of all words in special tags    
                            special_words_frequency[lemmatized] += 1 # aggregating frequency of all words in special tags
                
            for word in weight_dict.keys():                    
                document = collection.find_one({"token": word, "postingsList.docId": key}) # this is the retrieved frequency
                print(document)
                # word_ranking = (int(term_frequency) - special_words_frequency[word]) + weight_dict[word]
                # print("Word:", word, "Frequency:", term_frequency, "Special Frequency:", special_words_frequency[word], "Ranking:", word_ranking, "Doc ID:", key)
                # # collection.update_one({"token": word, "postingsList.docId": key}, {'$set': {"postingsList.$.tagScore": word_ranking}}) # this is the sample update

            if counter == 100:
                break

            # Formula for special score is (total frequency - total frequency of special words)*1 + special words * ranking factor
            

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
                # 'sentencePosition': index_data.position_sentence,
                # 'documentPosition': index_data.position_document,
                # 'ranking': index_data.ranking,
                'frequency': index_data.frequency
                }
            document['postingsList'].append(posting)
        documents.append(document)

    with open('inverted_index.json', 'w') as file:
        json.dump(documents, file)

    return documents
    


def get_from_db(phrase, query_vector = None):
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
    result = collection.find_one({'token': phrase})
    
    if result:
        postings = result['postingsList']
        ranked_postings = []
        for posting in postings:
            document_vector = np.array(list(posting['tf_idf'].values()))  # Convert TF-IDF dictionary to array
            similarity = cosine_similarity(query_vector, document_vector)
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

    for i in range(3):
        phrase = input("Enter a phrase to search for: ")
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

#### Page Rank
# Function to initialize PageRank scores
def initialize_pagerank(urls):
    num_urls = len(urls)
    return np.ones(num_urls) / num_urls

# Function to calculate PageRank scores
def calculate_pagerank(urls, adjacency_matrix, damping_factor=0.85, max_iterations=100, convergence_threshold=1e-5):
    num_urls = len(urls)
    pagerank = initialize_pagerank(urls)
    for _ in range(max_iterations):
        new_pagerank = np.ones(num_urls) * (1 - damping_factor) / num_urls + damping_factor * np.dot(adjacency_matrix.T, pagerank)
        if np.linalg.norm(new_pagerank - pagerank) < convergence_threshold:
            break
        pagerank = new_pagerank
    return pagerank

# Function to build the adjacency matrix from link structure
def build_adjacency_matrix(urls, links):
    num_urls = len(urls)
    adjacency_matrix = np.zeros((num_urls, num_urls))
    url_to_index = {url: index for index, url in enumerate(urls)}
    for source, destinations in links.items():
        if source in url_to_index:
            source_index = url_to_index[source]
            for destination in destinations:
                if destination in url_to_index:
                    destination_index = url_to_index[destination]
                    adjacency_matrix[destination_index, source_index] = 1
    return adjacency_matrix

# Function to normalize the adjacency matrix
def normalize_adjacency_matrix(adjacency_matrix):
    out_degree = np.sum(adjacency_matrix, axis=0)
    return adjacency_matrix / out_degree

# Function to get the list of URLs and links from the database
def fetch_urls_and_links():
    urls = []  # List of URLs
    links = {}  # Dictionary containing outgoing links for each URL
    # Fetch URLs and their outgoing links from the database
    # Replace the following lines with your MongoDB queries
    # Example: cursor = collection.find({}, {"_id": 0, "url": 1, "outgoing_links": 1})
    #          for document in cursor:
    #              urls.append(document['url'])
    #              links[document['url']] = document['outgoing_links']
    return urls, links

def compute_page_rank():
    # Fetch URLs and links from the database
    urls, links = fetch_urls_and_links()
    
    # Build the adjacency matrix from link structure
    adjacency_matrix = build_adjacency_matrix(urls, links)
    
    # Normalize the adjacency matrix
    adjacency_matrix = normalize_adjacency_matrix(adjacency_matrix)
    
    # Calculate PageRank scores
    pagerank = calculate_pagerank(urls, adjacency_matrix)
    

    urls, pagerank = compute_page_rank()
    for url, score in zip(urls, pagerank):
        print(f"URL: {url}, PageRank Score: {score}")
    
    return urls, pagerank
def cosine_similarity(query_vector, document_vector):
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
    query_vector = get_query()
    similarity = cosine_similarity(query_vector, document_vector)
    print("Cosine Similarity:", similarity)
    return query_vector




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
    query = query.strip().tolower()
    #get tf-idf of all words in query
    frequency = {}
    for i in range(len(query)-1):
        if query[i] not in stop_words:
            frequency[query[i]] += 1
            if query[i+1] not in stop_words:
                frequency[query[i]+" "+query[i+1]] += 1
    if query[len(query)-1] not in stop_words:
        frequency[query[len(query)-1]] += 1

    query_list=[]

    for key, value in frequency.values():
        result = collection.find_one({'token': key})
    
        if result:
            postings = result['postingsList']
            tf_idf = calculate_tf_idf(len(postings),value)
            query_list.append(tf_idf)
    return query_list  #returns a list

if __name__ == "__main__":
    # run_and_extract()
    # documents = prepare_documents_for_insertion(inverted_index)
    # calculate_ranking(inverted_index)
    # store_in_db(documents)
    # restore_db_from_json()
    # calculate_idf_from_mongo()
    add_special_words_to_index()
    # user_entries = get_top_entries()
    # output_analysis(user_entries)
=======
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

CORPUS_SIZE = 37000

class IndexData:
    def __init__(self, url): # pos_in_sentence=0, pos_in_doc=0, ranking=0
        self.url = url
        # self.position_sentence = [pos_in_sentence]
        # self.position_document = [pos_in_doc]
        #self.ranking = ranking
        self.tf_idf = 0
        self.frequency = 1  #in document
        
    # def add_pos_sentence(self, pos_sent):
    #     self.position_sentence.append(pos_sent)
    
    # def add_pos_doc(self, pos_doc):
    #     self.position_document.append(pos_doc)

    def increment_frequency(self):
        self.frequency += 1

    def __str__(self):
        return (f"URL: {self.url}, ")
                # f"Positions in Sentence: {self.position_sentence}, "
                # f"Positions in Document: {self.position_document}, "
                # f"Ranking: {self.ranking}, "
                # f"Frequency: {self.frequency}")
    
    def to_dict(self):
        return {
            'url': self.url,
        }
        #     'position_sentence': self.position_sentence,
        #     'position_document': self.position_document,
        #     'ranking': self.ranking,
        #     'frequency': self.frequency
        # }
    

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


# Initialize MongoDB client
client = pymongo.MongoClient("mongodb://localhost:27017/")
# Choose or create a database
db = client["search_engine"]
# Choose or create a collection
collection = db["inverted_index_test"] #changed to test

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

    if "inverted_index_test" in db.list_collection_names():
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

            # For testing small document size ##
            # counter += 1
            # if counter == 1000:
            #     print(f"Test size: {counter}")
                # for term, docs in inverted_index.items():
                #     safe_print(f"Term: {term}")
                #     for doc_id, index_data in docs.items():
                #         safe_print(f" Doc ID: {doc_id}, IndexData: {index_data}")    
                # break
            ## Feel free to comment out ##
                    


def extract_special_words(soup):
    special_words = {}
    relevance_tags = ['title', 'meta', 'h1', 'h2', 'h3', 'b', 'strong', 'a'] #title gets 4 points, meta/h1/h2/h3 gets 3, strong/b/a get 2, every other tag gets 1
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
            list_of_words.remove(word)
    return list_of_words
    #remove stop words

def normalize_word_list(list_of_words):
    for word in list_of_words:
        word = word.lower()
    return list_of_words


def index_word(word, position, url, pos_in_doc, ranking, key):
    result_analytics.update_word_count(word)
    result_analytics.add_to_docIds(key)
    if word not in inverted_index: #word hasn't appeared
        inverted_index[word] = {key: IndexData(url)} #{key: IndexData(url, position, pos_in_doc, ranking)} removing attributes to save space for now
    else:
        if key not in inverted_index[word]: #changed from list of dictionaries to nested dictionary
            inverted_index[word][key] = IndexData(url) # IndexData(url, position, pos_in_doc, ranking)
        else:
            # inverted_index[word][key].add_pos_sentence(position)
            # inverted_index[word][key].add_pos_doc(pos_in_doc)
            inverted_index[word][key].increment_frequency()
    
def create_index(text, key, url, special_words):
    if result_analytics.is_url_duplicate(url):
        return
    result_analytics.update_urls_discovered(url)
    
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
def store_in_db(documents): # storing inverted index into json file in case we lose information from the DB
    with open('error_log.txt', 'a') as file:
        with open('inverted_index.json', 'r') as inverted_index_file:
            data = json.load(inverted_index_file)
            for document in data:
                try:
                    collection.insert_one(document)
                except Exception as e:
                    file.write(f"{document['token']} too big to store: {len(document['postingsList']), {e}} \n")

        # for document in documents:
        #     try:
        #         collection.insert_one(document)
        #     except Exception as e:
        #         # Assuming 'token' is a key in your document
        #         # If 'token' might not be present, consider using document.get('token', 'unknown') to avoid KeyError
        #         file.write(f"{document['token']} too big to store: {document['postingsList']}\n")

    # collection.insert_many(documents)

def add_special_words_to_index(text, key, url, special_words): # special words per document
    documents = collection.find()
    with open(web_directory+"bookkeeping.json", 'r') as file:
        data = json.load(file)

        for key in data: 
            # Getting the text so that it can be tokenized, lemmatized, and indexed
            with open(web_directory+key, 'r', encoding='utf-8') as file:
                content = file.read()
                soup = BeautifulSoup(content, 'html.parser')
                special_words = extract_special_words(soup)

                collection.update_one({"token": "5 3", "postingsList.docId": "0/0"}, {'$set': {"postingsList.$.ranking": 5}}) # this is the sample update

                # # Idea is to tag special words
                lemmatizer = WordNetLemmatizer()
                word_frequency = defaultdict(int)
                for tag, words in special_words.items():
                    special_tokens = word_tokenize(words)
                    filtered_tokens = filter_words(special_tokens)
                    normalized_tokens = normalize_word_list(filtered_tokens)
                    tagged_words = pos_tag(normalized_tokens)
                    length = len(tagged_words)
                    for i in range(length):
                        word_net_pos = get_wordnet_pos(tagged_words[i][1])
                        lemmatized = lemmatizer.lemmatize(tagged_words[i][0], pos=word_net_pos)

                        if (tag == 'title'):
                            word_weight = 4
                        elif (tag == 'meta' or tag == 'h1' or tag == 'h2' or tag == 'h3'):
                            word_weight = 3
                        elif (tag == 'b' or tag == 'strong' or tag == 'a'):
                            word_weight = 2
                        word_frequency[lemmatized] += 1
                        term_frequency = collection.find_one({"token": lemmatized, "postingsList.docId": key})['frequency'] # this is the retrieved frequency


                    collection.update_one({"token": lemmatized, "postingsList.docId": key}, {'$set': {"postingsList.$.ranking": 5}}) # this is the sample update

                        # Formula for special score is (total frequency - total frequency of special words)*1 + special words * ranking factor
            

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
                # 'sentencePosition': index_data.position_sentence,
                # 'documentPosition': index_data.position_document,
                # 'ranking': index_data.ranking,
                'frequency': index_data.frequency
                }
            document['postingsList'].append(posting)
        documents.append(document)

    with open('inverted_index.json', 'w') as file:
        json.dump(documents, file)

    return documents
    


def get_from_db(phrase, query_vector = None):
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
    result = collection.find_one({'token': phrase})
    
    if result:
        postings = result['postingsList']
        ranked_postings = []
        for posting in postings:
            document_vector = np.array(list(posting['tf_idf'].values()))  # Convert TF-IDF dictionary to array
            similarity = cosine_similarity(query_vector, document_vector)
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

    for i in range(3):
        phrase = input("Enter a phrase to search for: ")
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

#### Page Rank
# Function to initialize PageRank scores
def initialize_pagerank(urls):
    num_urls = len(urls)
    return np.ones(num_urls) / num_urls

# Function to calculate PageRank scores
def calculate_pagerank(urls, adjacency_matrix, damping_factor=0.85, max_iterations=100, convergence_threshold=1e-5):
    num_urls = len(urls)
    pagerank = initialize_pagerank(urls)
    for _ in range(max_iterations):
        new_pagerank = np.ones(num_urls) * (1 - damping_factor) / num_urls + damping_factor * np.dot(adjacency_matrix.T, pagerank)
        if np.linalg.norm(new_pagerank - pagerank) < convergence_threshold:
            break
        pagerank = new_pagerank
    return pagerank

# Function to build the adjacency matrix from link structure
def build_adjacency_matrix(urls, links):
    num_urls = len(urls)
    adjacency_matrix = np.zeros((num_urls, num_urls))
    url_to_index = {url: index for index, url in enumerate(urls)}
    for source, destinations in links.items():
        if source in url_to_index:
            source_index = url_to_index[source]
            for destination in destinations:
                if destination in url_to_index:
                    destination_index = url_to_index[destination]
                    adjacency_matrix[destination_index, source_index] = 1
    return adjacency_matrix

# Function to normalize the adjacency matrix
def normalize_adjacency_matrix(adjacency_matrix):
    out_degree = np.sum(adjacency_matrix, axis=0)
    return adjacency_matrix / out_degree

# Function to get the list of URLs and links from the database
def fetch_urls_and_links():
    urls = []  # List of URLs
    links = {}  # Dictionary containing outgoing links for each URL
    # Fetch URLs and their outgoing links from the database
    # Replace the following lines with your MongoDB queries
    # Example: cursor = collection.find({}, {"_id": 0, "url": 1, "outgoing_links": 1})
    #          for document in cursor:
    #              urls.append(document['url'])
    #              links[document['url']] = document['outgoing_links']
    return urls, links

def compute_page_rank():
    # Fetch URLs and links from the database
    urls, links = fetch_urls_and_links()
    
    # Build the adjacency matrix from link structure
    adjacency_matrix = build_adjacency_matrix(urls, links)
    
    # Normalize the adjacency matrix
    adjacency_matrix = normalize_adjacency_matrix(adjacency_matrix)
    
    # Calculate PageRank scores
    pagerank = calculate_pagerank(urls, adjacency_matrix)
    

    urls, pagerank = compute_page_rank()
    for url, score in zip(urls, pagerank):
        print(f"URL: {url}, PageRank Score: {score}")
    
    return urls, pagerank
def cosine_similarity(query_vector, document_vector):
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
    query_vector = get_query()
    similarity = cosine_similarity(query_vector, document_vector)
    print("Cosine Similarity:", similarity)
    return query_vector




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
    query = query.strip().tolower()
    #get tf-idf of all words in query
    frequency = {}
    for i in range(len(query)-1):
        if query[i] not in stop_words:
            frequency[query[i]] += 1
            if query[i+1] not in stop_words:
                frequency[query[i]+" "+query[i+1]] += 1
    if query[len(query)-1] not in stop_words:
        frequency[query[len(query)-1]] += 1

    query_list=[]

    for key, value in frequency.values():
        result = collection.find_one({'token': key})
    
        if result:
            postings = result['postingsList']
            tf_idf = calculate_tf_idf(len(postings),value)
            query_list.append(tf_idf)
    return query_list  #returns a list

if __name__ == "__main__":
    # run_and_extract()
    # documents = prepare_documents_for_insertion(inverted_index)
    # calculate_ranking(inverted_index)
    # store_in_db(documents)
    # restore_db_from_json()
    # calculate_idf_from_mongo()
    # add_special_words_to_index()
    # user_entries = get_top_entries()
    # output_analysis(user_entries)
>>>>>>> 5775b31b12ebad32b39c6b30e8fa57f3b17d97b2

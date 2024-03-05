# Script for running through webpages and extracting the html content
import json
from bs4 import BeautifulSoup
from collections import defaultdict
import pymongo
from urllib.parse import urljoin
from analytics import Analytics
from nltk import sent_tokenize, WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from utility import tag_words, get_wordnet_pos, safe_print

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
    
        
web_directory = 'webpages/WEBPAGES_RAW/'
inverted_index = defaultdict()
anchor_urls = defaultdict(list)
result_analytics = Analytics()

def run_and_extract():
    """
    Reads input from bookkeeping.json, locates each file, and attempts to parse each document
    """

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
    


def create_index(text, key, url, special_words):
    if result_analytics.is_url_duplicate(url):
        return
    result_analytics.update_urls_discovered(url)

    
    list_of_sent = sent_tokenize(text) #list of sentences

    lemmatizer = WordNetLemmatizer()

    for sent in list_of_sent:
        tagged_words = tag_words(sent)
        length = len(tagged_words)
        for i in range(length):
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


def get_inverted_index():
    return inverted_index

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
import networkx as nx

class InvertedIndex:
    def __init__(self):
        self.inverted_index = defaultdict()
        self.anchor_urls = defaultdict(list)
        self.result_analytics = Analytics()
        self.html_content = defaultdict(str)
        self.outgoing_links = defaultdict(set)   # PAGERANK - Stores links between pages
        self.incoming_links = defaultdict(set)   # PAGERANK - Stores links between pages
        self.web_directory = 'webpages/WEBPAGES_RAW/'
        
        #self.outgoing_links = defaultdict(set)   # PAGERANK - Stores links between pages
        #self.incoming_links = defaultdict(set)   # PAGERANK - Stores links between pages
        self.links = defaultdict(set)

    def run_and_extract(self):
        """
        Reads input from bookkeeping.json, locates each file, and attempts to parse each document
        """
   
        with open(self.web_directory+"bookkeeping.json", 'r') as file:
            data = json.load(file)

            counter = 0
            for key in data: 
                # Getting the text so that it can be tokenized, lemmatized, and indexed
                with open(self.web_directory+key, 'r', encoding='utf-8') as file:
                    content = file.read()
                    soup = BeautifulSoup(content, 'html.parser')

                    special_words = self.extract_special_words(soup, data[key])
                    text = soup.get_text()

                    if data[key] in self.anchor_urls:
                        text += " ".join(self.anchor_urls[data[key]])

                    #Passes the parsed HTML to create_index
                    self.create_index(text, key, data[key], special_words)
                    
            # For testing small document size ##
                # counter += 1
                # if counter == 1000:
                #     # print(f"Test size: {counter}")
                #     # for term, docs in self.inverted_index.items():
                #     #     safe_print(f"Term: {term}")
                #     #     for doc_id, index_data in docs.items():
                #     #         safe_print(f" Doc ID: {doc_id}, IndexData: {index_data}")
                #     break
                # Feel free to comment out ##
        # self.result_analytics.output_analysis()
                        
    def get_page_rank_urls(self):
        """ Runs through the corpus again, this time extracting pagerank """

        self.incoming_links.clear()
        self.outgoing_links.clear()

        g=nx.DiGraph()

        with open(self.web_directory+"bookkeeping.json", 'r') as file:
            data = json.load(file)
            counter = 0
            for key in data: 
                with open(self.web_directory+key, 'r', encoding='utf-8') as file:
                    content = file.read()
                    soup = BeautifulSoup(content, 'html.parser')
                    self.outgoing_links[data[key]] = set()
                    self.incoming_links[data[key]] = set()
                    # PAGERANK - Extract and store links
                    for url in soup.find_all('a'):
                        if url.get('href') is not None:
                            link = urljoin(data[key], url.get('href'))
                            self.outgoing_links[data[key]].add(link) 
                            self.incoming_links[link].add(data[key])
        
            for root_url, outlinks in self.outgoing_links.items():
                g.add_node(root_url)
                for outlink in outlinks:
                    g.add_edge(root_url, outlink)
            
        pagerank = nx.pagerank(g, alpha=0.85, personalization=None, weight='weight', dangling=None)
        return pagerank
 
    def sort_by_page_rank(self, pageranks):
        sorted_pageranks = sorted(pageranks.items(), key=lambda item: item[1], reverse=True)

        for url, pagerank in sorted_pageranks:
            print("URL: ", url, "PageRank: ", pagerank, "\n")
        #sorted_pageranks_dict = dict(sorted_pageranks)
        # print(f'sorted_pageranks = {sorted_pageranks}')
        return sorted_pageranks
        # Format is:  [(First Highest Page Rank URL, PageRank), (Second Highest Page Rank URL, PageRank), ...]
        
        
        
        # return (self.incoming_links, self.outgoing_links)
        #pagerank = nx.pagerank_numpy(g, alpha=0.85, personalization=None, weight='weight', dangling=None)
        
  
                        
    def extract_special_words(self, soup, url):
        special_words = {}
        relevance_tags = ['title', 'meta', 'h1', 'h2', 'h3', 'b', 'strong', 'a']
        for tag in relevance_tags:
            for match in soup.find_all(tag): # find all the words that are in important tags, need later for positional index retrieval and ranking
                matched_text = match.get_text()
                if matched_text.strip() == "": #skip empty tags
                    continue
                if tag == 'a':
                    if match.get('href') is not None:
                        self.anchor_urls[urljoin(url, match.get('href'))].append(match.get_text())
                if tag in special_words:
                    special_words[tag].append(match.get_text())
                else:
                    special_words[tag] = [match.get_text()]
                match.decompose() # removes tag from tree
        return special_words
        


    def index_word(self, word, url, key, tag):
        self.result_analytics.update_word_count(word)
        self.result_analytics.add_to_docIds(key)
        if word not in self.inverted_index: #word hasn't appeared
            self.inverted_index[word] = {key: IndexData(url)} #{key: IndexData(url, position, pos_in_doc, ranking)} removing attributes to save space for now
        else:
            if key not in self.inverted_index[word]: #changed from list of dictionaries to nested dictionary
                self.inverted_index[word][key] = IndexData(url) # IndexData(url, position, pos_in_doc, ranking)
            else:
                self.inverted_index[word][key].increment_frequency()
        self.inverted_index[word][key].add_tag_score(tag)
        


    def create_index(self, text, key, url, special_words):
        if self.result_analytics.is_url_duplicate(url):
            return
        self.result_analytics.update_urls_discovered(url)

        
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
                    self.index_word(two_gram, url, key, "p")
                    
                #store this into the DB along with i
                self.index_word(lemmatized, url, key, "p")      

        for tag, sent in special_words.items():
            for words in sent:
                tagged_words = tag_words(words)
                length = len(tagged_words)
                for i in range(length):
                    word_net_pos = get_wordnet_pos(tagged_words[i][1])
                    lemmatized = lemmatizer.lemmatize(tagged_words[i][0], pos=word_net_pos)
                    if(i < length -1):
                        two_gram = lemmatized + " " + lemmatizer.lemmatize(tagged_words[i+1][0], get_wordnet_pos(tagged_words[i+1][1]))
                        self.index_word(two_gram, url, key, tag)
                    self.index_word(lemmatized, url, key, tag)

    def get_inverted_index(self):
        return self.inverted_index

    def get_html_content(self):
        return self.html_content


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
    
    



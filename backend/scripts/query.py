# Query.py
# File for querying MongoDB and retrieving search results with the highest cosine-similarity + tag score

import json
import math
from collections import defaultdict

from bs4 import BeautifulSoup
import requests
from utility import tag_words, get_wordnet_pos, calculate_tf_idf
from nltk.stem import WordNetLemmatizer
from mongo import MongoDBClient
import runner
from flask import Flask, request, jsonify
#from scripts.query import search_query

def get_query_with_input():
    query = input("Enter what we need to search for: ")
    #sanitize query
    query = tag_words(query)
    #get tf-idf of all words in query
    lemmatizer = WordNetLemmatizer()
    query_len = len(query)
    query_terms = []
    for i in range(query_len - 1):
        lemmatized_word = lemmatizer.lemmatize(query[i][0], get_wordnet_pos(query[i][1]))
        two_gram = lemmatizer.lemmatize(query[i][0], get_wordnet_pos(query[i][1])) + " " + lemmatizer.lemmatize(query[i + 1][0], get_wordnet_pos(query[i + 1][1]))
        query_terms.append(two_gram)
        query_terms.append(lemmatized_word)
    lemmatized_word = lemmatizer.lemmatize(query[query_len - 1][0], get_wordnet_pos(query[query_len - 1][1]))
    query_terms.append(lemmatized_word)
    return query_terms
def get_query(query):
    print(f'get_query({query})')
    #sanitize query
    query = tag_words(query)
    #get tf-idf of all words in query
    lemmatizer = WordNetLemmatizer()
    query_len = len(query)
    query_terms = []
    for i in range(query_len - 1):
        lemmatized_word = lemmatizer.lemmatize(query[i][0], get_wordnet_pos(query[i][1]))
        two_gram = lemmatizer.lemmatize(query[i][0], get_wordnet_pos(query[i][1])) + " " + lemmatizer.lemmatize(query[i + 1][0], get_wordnet_pos(query[i + 1][1]))
        query_terms.append(two_gram)
        query_terms.append(lemmatized_word)
    lemmatized_word = lemmatizer.lemmatize(query[query_len - 1][0], get_wordnet_pos(query[query_len - 1][1]))
    query_terms.append(lemmatized_word)
    return query_terms

retrieved_urls = set()

def cosine_sim_and_scoring(query):
    print(f'cosine_sim_and_scoring({query})')
    retrieved_urls.clear()
    # Given a query, calculates the cosine similarity of it and the documents

    mongo_db_client = MongoDBClient()
    page_collection = MongoDBClient("page_rank").collection
    scores = defaultdict(float)
    magnitudeDoc = defaultdict(float)
    magnitudeQuery = 0
    url_list = defaultdict(str)
    tag_score = defaultdict(float)
    page_rank = defaultdict(float)
    results = []
    query_freq = defaultdict(int)
    title_desc = defaultdict(tuple)
    for word in query:
        query_freq[word] += 1
    
    for query_term in set(query):    #"query" being the words in the input.
        token = mongo_db_client.collection.find_one({'token': query_term})
        if token is None:
            continue
        query_tf_idf = calculate_tf_idf(query_freq[query_term], len(token['postingsList']))
        magnitudeQuery += query_tf_idf ** 2
        for posting in token['postingsList']:
            scores[posting['docId']] += posting['tf_idf'] * query_tf_idf
            magnitudeDoc[posting['docId']] += posting['tf_idf'] ** 2
            url_list[posting['docId']] = posting['url']
            tag_score[posting['docId']] = posting['tagScore']
            page_rank[posting['docId']] = posting['page_rank']
            title_desc[posting['docId']] = posting['title']
            
            if posting['url'] not in retrieved_urls:
                retrieved_urls.add(posting['url'])

    #normalize vectors
    #Equation for cosine similarity: A*B / |A|*|B|
    
    for docID, score in scores.items():
        cosine_score = score / (math.sqrt(magnitudeDoc[docID]) * math.sqrt(magnitudeQuery))
        total_score = (cosine_score*3.0) + (tag_score[docID]/3.0) + (page_rank[docID] * 10000) #  feel free to change the weights
        results.append((docID, url_list[docID], cosine_score*3.0, total_score, page_rank[docID] * 10000, tag_score[docID]/3.0, title_desc[docID][0], title_desc[docID][1]))
        #print(f'cosine: {cosine_score*10.0},  log of tag_score: {tag_score[docID]/10.0},  page_rank X 100000: {page_rank[docID] * 25000},  total_score: {total_score},  url: {url_list[docID]}')
    results.sort(key=lambda x: x[3], reverse=True)
    
    
    
    return results


def return_results(query):
    query_terms = get_query(query)
    #print(query_terms)
    results = cosine_sim_and_scoring(query_terms)
    print(f'results = {results}')
    
    top_20_results = results[:20]
    return top_20_results
    #return top_20_results
    # first_20_results = []
    
    # web_directory = 'webpages/WEBPAGES_RAW/'
    # for each_result in top_20_results:
         #url = each_result[1]
    #     print(f'url = {url}')
        
        #  try:
        #      #response = requests.get(url) 
        #     with open(web_directory+"bookkeeping.json", 'r') as file:
        #         data = json.load(file) 
        #         for key in data: 
        #             if key == each_result[0]:
        #                 # Getting the text so that it can be tokenized, lemmatized, and indexed
        #                 with open(web_directory+key, 'r', encoding='utf-8') as file:
        #                     content = file.read()
        #                     soup = BeautifulSoup(content, 'html.parser')

        #                     #extract_special_words(soup, data[key])
        #                     #text = soup.get_text()
                            
                            
        #                     title = soup.title.string if soup.title else 'No title found'
                            
        #                     print("title is: " + title)
                            
        #                 #     # Extract description (often found in a meta tag)
        #                     description_tag = soup.find('meta', attrs={'name': 'description'})
        #                     description = description_tag['content'] if description_tag else 'No description found'
        #                     first_20_results.append((each_result[0], each_result[1], each_result[2], each_result[3], each_result[4], each_result[5], title, description))
                    
        #                     #if data[key] in self.anchor_urls:
        #                     #    text += " ".join(self.anchor_urls[data[key]])

        #                     #Passes the parsed HTML to create_index
        #                     #self.create_index(text, key, data[key], special_words)
        #  except:
        #      print("error")
        #      continue
        
        
        
        
    '''  # Raise an exception if the request was unsuccessful
         #response.raise_for_status()
         soup = BeautifulSoup(response.text, 'html.parser')
         # Extract title
         title = soup.title.string if soup.title else 'No title found'
         print("title is: " + title)
        
    #     # Extract description (often found in a meta tag)
         description_tag = soup.find('meta', attrs={'name': 'description'})
         description = description_tag['content'] if description_tag else 'No description found'
         first_20_results.append(each_result[0], each_result[1], each_result[2], each_result[3], each_result[4], each_result[5], title, description)
         #print(f'each_result = {each_result}')'''
    
    return first_20_results
    

'''def get_search_results(query):
    print(f"query.py get_search_results({query})")

    results= [(1,600,'bing.com'),(2,600,'yahoo.com'),(2,600,'aol.com'), (2,600,'ebay.com')]
    first_20_results = results[:3]  # Assuming results is a list
    return first_20_results'''
    
if __name__ == "__main__":
    pass
    #app.run()
    #print("query.py __main__")
    #app.run(debug=True)
    
    #query_terms = get_query()
    #results = cosine_sim_and_scoring(query_terms)
    #first_30_results = results[:30]
    
    
    
    #print(results[0:3])
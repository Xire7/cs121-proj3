# Query.py
# File for querying MongoDB and retrieving search results with the highest cosine-similarity + tag score

import math
from collections import defaultdict
from utility import tag_words, get_wordnet_pos, calculate_tf_idf
from nltk.stem import WordNetLemmatizer
from mongo import MongoDBClient
import runner


def get_query():
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

retrieved_urls = set()

def cosine_sim_and_scoring(query):
    retrieved_urls.clear()
    # Given a query, calculates the cosine similarity of it and the documents

    mongo_db_client = MongoDBClient()
    scores = defaultdict(float)
    magnitudeDoc = defaultdict(float)
    magnitudeQuery = 0
    url_list = defaultdict(str)
    tag_score = defaultdict(float)
    results = []
    query_freq = defaultdict(int)
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
            
            if posting['url'] not in retrieved_urls:
                retrieved_urls.add(posting['url'])

    #normalize vectors
    #Equation for cosine similarity: A*B / |A|*|B|
    
    for docID, score in scores.items():
        cosine_score = score / (math.sqrt(magnitudeDoc[docID]) * math.sqrt(magnitudeQuery))
        total_score = (0.7*cosine_score) + (0.3*tag_score[docID])
        results.append((docID, cosine_score, url_list[docID], total_score))
    results.sort(key=lambda x: x[3], reverse=True)
    return results



if __name__ == "__main__":
    query_terms = get_query()
    print(query_terms)
    results = cosine_sim_and_scoring(query_terms)
    print(results)
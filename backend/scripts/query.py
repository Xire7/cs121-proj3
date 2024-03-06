# Query.py
# File for querying MongoDB and retrieving search results with the highest cosine-similarity + tag score

import math
from collections import defaultdict
from utility import tag_words, get_wordnet_pos, calculate_tf_idf, calculate_query_tf_idf
from nltk.stem import WordNetLemmatizer
from mongo import MongoDBClient


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
        query_terms.append(lemmatized_word)
    lemmatized_word = lemmatizer.lemmatize(query[query_len - 1][0], get_wordnet_pos(query[query_len - 1][1]))
    query_terms.append(lemmatized_word)

    return query_terms


def scoring(query):
    mongo_db_client = MongoDBClient()
    scores = defaultdict(float)
    magnitudeDoc = defaultdict(float)
    magnitudeQuery = defaultdict(float)
    url_list = defaultdict(str)
    tag_score = defaultdict(float)
    cosine_similarity_scores = []
    
    print("Sanitized query:", query)
    for query_term in query:    #"query" being the words in the input.
        token = mongo_db_client.collection.find_one({'token': query_term})
        query_tf_idf = calculate_query_tf_idf(query.count(query_term), len(token['postingsList']))
        for posting in token['postingsList']:
            scores[posting['docId']] += posting['tf_idf'] * query_tf_idf
            magnitudeDoc[posting['docId']] += posting['tf_idf'] ** 2
            magnitudeQuery[posting['docId']] += query_tf_idf ** 2
            url_list[posting['docId']] = posting['url']
            tag_score[posting['docId']] = posting['tagScore']

    #normalize vectors
    #Equation for cosine similarity: A*B / |A|*|B|
    for docID, score in scores.items():
        cosine_score = score / math.sqrt(magnitudeDoc[docID] * magnitudeQuery[docID]) 
        # print(f"DocID: {docID}, Cosine Similarity: {cosine_score}, URL: {url_list[docID]}, Dot Product: {score}, Query Length: {magnitudeQuery[docID]}, Document Length: {magnitudeDoc[docID]}, Normalized Length: {math.sqrt(magnitudeDoc[docID] * magnitudeQuery[docID])}")
        total_score = (0.8*cosine_score) + (0.2*tag_score[docID])
        cosine_similarity_scores.append((docID, cosine_score, url_list[docID], total_score))
    cosine_similarity_scores.sort(key=lambda x: x[1], reverse=True)
    return cosine_similarity_scores



if __name__ == "__main__":
    query_terms = get_query()
    results = scoring(query_terms)
    # print(results)
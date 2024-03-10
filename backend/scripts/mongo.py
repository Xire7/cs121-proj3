# Mongo.py
# File for interacting and manipulating the MongoDB database

import pymongo
from utility import calculate_tf_idf, safe_print
import ijson
from collections import defaultdict

class MongoDBClient:
    def __init__(self, db_name="inverted_index_normalized"):
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["search_engine"]
        self.collection = self.db[db_name]

def calculate_idf_from_mongo():
    collection = MongoDBClient().collection
    documents = collection.find()
    for document in documents:
        postings = document['postingsList'] # postings is the List of Objects
        for posting in postings:
            posting_value = calculate_tf_idf(len(postings), posting['frequency'])
            posting['tf_idf'] = posting_value
        collection.update_one({'_id': document['_id']}, {'$set': {'postingsList': postings}})
        print("Updated document", document['token'])
    return

def calculate_page_rank_from_mongo():
    inverted_index = MongoDBClient().collection
    documents = inverted_index.find()
    page_ranks = MongoDBClient("page_rank").collection
    pages = page_ranks.find()
    page_rank = defaultdict(float)

    counter = 0
    for doc in pages:
        page_rank[doc['url']] = doc['pageRank']
        counter += 1
        print(counter, doc['url'])

    for document in documents:
        postings = document['postingsList']
        for posting in postings:
            posting['page_rank'] = page_rank[posting['url']]
        inverted_index.update_one({'_id': document['_id']}, {'$set': {'postingsList': postings}})
        print("Updated document", document['token'])

def restore_db_from_json():
    collection = MongoDBClient().collection
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


def prepare_documents_for_insertion(inverted_index):
    mongoDBClient = MongoDBClient()

    if "inverted_index_normalized" in mongoDBClient.db.list_collection_names():
        safe_print("Collection exists, dropping again...")
        mongoDBClient.collection.drop()
        safe_print("Collection 'inverted_index_normalized' dropped.")
        
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
    mongoDBClient.collection.insert_many(documents)

    return documents


def create_page_rank_collection(page_ranks):
    mongoDBClient = MongoDBClient("page_rank")
    if "page_rank" in mongoDBClient.db.list_collection_names():
        safe_print("Collection exists, dropping again...")
        mongoDBClient.db.drop_collection("page_rank")
        safe_print("Collection 'page_rank' dropped.")
    documents = []
    for url, rank in page_ranks.items():
        document = {
            'url': url,
            'pageRank': rank
        }
        documents.append(document)
    mongoDBClient.collection.insert_many(documents)
    return documents


if __name__ == "__main__":
    calculate_idf_from_mongo()
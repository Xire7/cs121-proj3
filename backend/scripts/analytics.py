#Analytics.py
# File for writing the analytics to a file

from mongo import MongoDBClient

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
    
    def output_analysis(self):
        with open('additional_analytics.txt','w', encoding='utf-8') as file:
            # for query in query_result_list:
            #     if query[1] is not None:
            #         file.write(f"Total number of links for phrase {query[0]}: {query[1][1]} \n")
            #         file.write(f"Top 20 entries for phrase '{query[0]}':\n")
            #         for i, entry in enumerate(query[1][0], 1):
            #             # file.write(f"{i}. Doc ID: {entry['docId']}, URL: {entry['url']}, Sentence Position: {entry['sentencePosition']}, Document Position: {entry['documentPosition']}, Ranking: {entry['ranking']}, Frequency: {entry['frequency']}\n")
            #             file.write(f"{i}. Doc ID: {entry['docId']}, URL: {entry['url']} Frequency: {entry['frequency']}\n")

            #     else:
            #         file.write(f"No entries found for word '{query[0]}'.\n")


            file.write('\nAdditional analytics: \n')
            file.write(f"Number of Unique Words: {self.word_count}\n")
            file.write(f'URL Count = {self.url_count}\n')
            file.write(f'Number of unique documents = {len(self.unique_docIds)} \n')
            collection_stats = MongoDBClient().db.command("collStats", "inverted_index_normalized")

            # Extract the size from the stats
            collection_size_in_bytes = collection_stats['size']
            file.write(f'Database Size = {collection_size_in_bytes} KB')


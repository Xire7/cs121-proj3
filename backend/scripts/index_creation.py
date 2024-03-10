# index_creation.py
# Purpose is to run the corpus and create the inverted index to store in the DB

import runner, mongo, page_rank, query


def main():
    inverted_index = runner.InvertedIndex()
    inverted_index.run_and_extract()
    
    
    #page_rank.compute_page_rank(inverted_index.links)
    
    

    pagerank_rankings = page_rank.page_rank_rankings(inverted_index.incoming_links, inverted_index.outgoing_links) # **** "results" represents the list of search results obtained from a search query, so pass that in after processing user input
    
    # ^^^^ THIS RETURNS DICT where keys are the urls and values are the pagerank scores.
    
    
    # mongo.prepare_documents_for_insertion(inverted_index.inverted_index)
    # mongo.calculate_idf_from_mongo()

if __name__ == "__main__":
    main()

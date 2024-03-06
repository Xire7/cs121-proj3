# index_creation.py
# Purpose is to run the corpus and create the inverted index to store in the DB

import runner, mongo


def main():
    # inverted_index = runner.InvertedIndex()
    # inverted_index.run_and_extract()
    # mongo.prepare_documents_for_insertion(inverted_index.inverted_index)
    mongo.calculate_idf_from_mongo()

if __name__ == "__main__":
    main()

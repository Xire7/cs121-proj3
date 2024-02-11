# Script for running through webpages and extracting the html content
import json
import cbor2
from cbor import load
from bs4 import BeautifulSoup
import math

web_directory = 'webpages/WEBPAGES_RAW/'


"""
    Reads input from bookkeeping.json, locates each file, and attempts to parse each document
"""
def run_and_extract():
    with open(web_directory+"bookkeeping.json", 'r') as file:
        data = json.load(file)
        for key in data: 
            with open(web_directory+key, 'r', encoding='utf-8') as file:
                print("KEY:", key, "URL:", data[key])
                content = file.read()
                soup = BeautifulSoup(content, 'html.parser')
                print(soup.get_text().encode('utf-8', errors='ignore'))
                #call tokenize function

def create_index():
    pass



def calculate_ranking(term, numOfDocs, termFrequency, docWordCount):
    # term = current word to calculate ranking of
    # numOfDocs (N) = Number of documents containing the term t
    # termFrequency (df) = frequency of term t in ENTIRE corpus
    # docWordCount (tf) = df / word count of ENTIRE corpus
    
    '''
    * tf-idf(t, d) = tf(t, d) * idf(t)
        N(t) = Number of documents containing the term t
        df(t) = occurrence of t in documents
    * tf(t,d) = count of t in d / number of words in d
    * idf(t) = log(N/ df(t))

    '''

    
    # numOfDocs (N) = Number of documents containing the term t
    # termFrequency (df) = frequency of term t in ENTIRE corpus
    # docWordCount (tf) = df / word count of ENTIRE corpus
    tf = termFrequency / docWordCount
    idf = math.log(numOfDocs / termFrequency)
    tf_idf = tf * idf

    print(f"TF-IDF of '{term}: {str(tf_idf)}'")
    
    #tf_idf = tf * idf

    pass


if __name__ == "__main__":
    run_and_extract()
    #create_index()
    #calculate_ranking()
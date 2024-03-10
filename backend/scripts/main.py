# Main engine program to query and fetch search results

import query

def main(words):
    #return [(1,600,'bing.com'),(2,600,'ebay.com.com'),(2,600,'ebay.com.com'),(2,600,'ebay.com.com'),(2,600,'ebay.com.com'),(2,600,'ebay.com.com'),(2,600,'ebay.com.com'),(2,600,'ebay.com.com'),(2,600,'ebay.com.com'),(2,600,'ebay.com.com')
    #        ,(2,600,'ebay.com.com'),(2,600,'ebay.com.com'),(2,600,'ebay.com.com'),(2,600,'ebay.com.com'),(2,600,'ebay.com.com'),(2,600,'ebay.com.com')]
    print(f'main.py')
    
    result = query.return_results(words)
    print(f'~Result~  =  {result}')
    return result

# (doc_id, score, url)
# if __name__ == "__main__":
#     main()
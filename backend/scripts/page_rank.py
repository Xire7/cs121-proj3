import numpy as np
import query
# Function to get the list of URLs and links from the database


# PAGERANK

'''def compute_page_rank(links, d=0.85, max_iterations=100, tol=1e-6):
    pages = list(links.keys())
    print(f'pages = {pages}' )
    num_pages = len(pages)
    if num_pages == 0:
        return {}
        
    page_rank = dict.fromkeys(pages, 1 / num_pages)
    print(f'page_rank = {page_rank}' )
    new_rank = dict.fromkeys(pages, 0)
    print(f'new_rank = {new_rank}' )

    for _ in range(max_iterations):
        total_change = 0
        for page in pages:
            
            rank_sum = 0
            for link in links[page]:
                if link in page_rank:
                    if len(links[link])>0:
                        rank_sum += (page_rank[link] / len(links[link]))
                    else:
                        rank_sum += 0

            print(f'rank_sum = {rank_sum}')

            #rank_sum = sum(page_rank[link] / len(links[link]) for link in links[page] if link in page_rank)
            one = 1-d
            two = d * sum(  PR(eachLinkThatPointsToIt)/NumOfLinksGoingOutOfThisLink  )
            page_rank_recursive()
            new_rank_value = (1 - d) / num_pages + d * rank_sum
            total_change += abs(page_rank[page] - new_rank_value)
            new_rank[page] = new_rank_value
        
        page_rank, new_rank = new_rank, page_rank
        if total_change < tol:
            break

    print(f'NEW page_rank = {page_rank}' )
    print(f'NEW new_rank = {new_rank}' )
    return page_rank'''

def compute_outgoing_links(links):
    # Calculate the number of outgoing links for each page
    outgoing_links_count = {}
    for page in links:
        # If a page has outgoing links, count them; otherwise, the page has no outgoing links
        outgoing_links_count[page] = len(links[page])
    return outgoing_links_count


def pagerank(inlinks, outlinks, d=0.85, convergence_threshold=0.0001, max_iterations=1, pageranks=None):
    if pageranks is None:
        #pageranks = {page: 1.0 / len(inlinks) for page in inlinks}
        pageranks = {page: 1.0 - d for page in inlinks}
    num_pages = len(inlinks)
    #for iteration in range(max_iterations):
    num_of_interations=0

    while True:
        num_of_interations +=1
        if (num_of_interations>max_iterations):
            break
        
        new_pageranks = {}
        total_change = 0

        total_pagerank = 0

        for page in pageranks:
            if page in inlinks and inlinks[page]:
                new_pagerank = 0
                for inlink in inlinks[page]:
                    if inlink in pageranks and inlink in outlinks and outlinks[inlink]:
                        #print("")
                        new_pagerank += pageranks[inlink] / len(outlinks[inlink])
                    else:
                        new_pagerank += pageranks[inlink] / len(inlinks)
                new_pagerank = (1 - d) + d * new_pagerank
                total_change += abs(new_pagerank - pageranks[page])
                new_pageranks[page] = new_pagerank
            
                total_pagerank += new_pagerank
            
            else:
                new_pageranks[page] = pageranks[page]  # Page with no inlinks retains its PageRank
        
        pageranks = new_pageranks
        
        #if total_change < convergence_threshold:
        #    break
        average_pagerank = total_pagerank / num_pages
        
        print(f'average_pagerank = {average_pagerank}' )
        if abs(average_pagerank - 1.0) < convergence_threshold:
            break
    print(f'pageranks = {pageranks}')

    # Calculate the sum of all PageRank values
    sum_pageranks = sum(pageranks.values())

    # Normalize the PageRank values
    normalized_pageranks = {page: pr / sum_pageranks for page, pr in pageranks.items()}
    print(f'normalized_pageranks (sum={sum(normalized_pageranks.values())}) = {normalized_pageranks}')

    #return pageranks
    return normalized_pageranks

def page_rank_rankings(incoming_links, outgoing_links):
    desired_incoming_links = {key: value for key, value in incoming_links.items() if key in query.retrieved_urls}
    normalized_pageranks = pagerank(desired_incoming_links, outgoing_links)
    
    # Sort the pages by their PageRank values
    sorted_pageranks = sorted(normalized_pageranks.items(), key=lambda x: x[1], reverse=True)
    return sorted_pageranks


if __name__ == "__main__":
    example_self_links = dict([('linkA.com', ['linkFromA1.com', 'linkFromA2.com', 'linkFromA3.com', 'linkB.com', 'linkC.com']),
                         ('linkB.com', ['linkFromB1.com', 'linkFromB2.com', 'linkC.com', 'linkC.com', 'linkC.com', 'linkC.com', 'linkC.com', 'linkC.com', 'linkC.com',]),
                         ('linkC.com', ['linkFromC1.com'])
                         ])
    example_self_links1 = dict([('PageA', ['PageB', 'PageC']),
                         ('PageB', ['PageC']),
                         ('PageC', ['PageA']),
                         ('PageD', ['PageC'])
                         ])
    example_self_links2 = dict([('PageA', ['PageB', 'PageC']),
                         ('PageB', ['PageC']),
                         ('PageC', []),
                         ])
    example_self_links3 = dict([('PageA', []),
                         ('PageB', ['PageA']),
                         ('PageC', ['PageA','PageB']),
                         ])
    #compute_page_rank(example_self_links3, example_self_links2, 'PageA')
    #agerank("PageA", example_self_links3, example_self_links2)
    #pagerank("PageB", example_self_links3, example_self_links2)
    pagerank("PageC", example_self_links3, example_self_links2)




'''def compute_page_rank():
    # Fetch URLs and links from the database
    urls, links = fetch_urls_and_links()
    
    # Build the adjacency matrix from link structure
    adjacency_matrix = build_adjacency_matrix(urls, links)
    
    # Normalize the adjacency matrix
    adjacency_matrix = normalize_adjacency_matrix(adjacency_matrix)
    
    # Calculate PageRank scores
    pagerank = calculate_pagerank(urls, adjacency_matrix)
    

    urls, pagerank = compute_page_rank()
    for url, score in zip(urls, pagerank):
        print(f"URL: {url}, PageRank Score: {score}")
    
    return urls, pagerank



def fetch_urls_and_links():
    urls = []  # List of URLs
    links = {}  # Dictionary containing outgoing links for each URL
    # Fetch URLs and their outgoing links from the database
    # Replace the following lines with your MongoDB queries
    # Example: cursor = collection.find({}, {"_id": 0, "url": 1, "outgoing_links": 1})
    #          for document in cursor:
    #              urls.append(document['url'])
    #              links[document['url']] = document['outgoing_links']
    return urls, links


# Function to build the adjacency matrix from link structure
def build_adjacency_matrix(urls, links):
    num_urls = len(urls)
    adjacency_matrix = np.zeros((num_urls, num_urls))
    url_to_index = {url: index for index, url in enumerate(urls)}
    for source, destinations in links.items():
        if source in url_to_index:
            source_index = url_to_index[source]
            for destination in destinations:
                if destination in url_to_index:
                    destination_index = url_to_index[destination]
                    adjacency_matrix[destination_index, source_index] = 1
    return adjacency_matrix


# Function to normalize the adjacency matrix
def normalize_adjacency_matrix(adjacency_matrix):
    out_degree = np.sum(adjacency_matrix, axis=0)
    return adjacency_matrix / out_degree


#### Page Rank
# Function to initialize PageRank scores
def initialize_pagerank(urls):
    num_urls = len(urls)
    return np.ones(num_urls) / num_urls

# Function to calculate PageRank scores
def calculate_pagerank(urls, adjacency_matrix, damping_factor=0.85, max_iterations=100, convergence_threshold=1e-5):
    num_urls = len(urls)
    pagerank = initialize_pagerank(urls)
    for _ in range(max_iterations):
        new_pagerank = np.ones(num_urls) * (1 - damping_factor) / num_urls + damping_factor * np.dot(adjacency_matrix.T, pagerank)
        if np.linalg.norm(new_pagerank - pagerank) < convergence_threshold:
            break
        pagerank = new_pagerank
    return pagerank'''

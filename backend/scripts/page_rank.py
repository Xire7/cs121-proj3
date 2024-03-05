
# Function to get the list of URLs and links from the database

def compute_page_rank():
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
    return pagerank

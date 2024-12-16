import json
import numpy as np

def load_data_1(file_path):
    """
    Load the JSON file and return the characters and interactions
    IN: file_path, str, path to the JSON files
    OUT: list of dict, characters
         list of tuple, interactions
    """
    with open(file_path, "r") as file:
        data = json.load(file)

    # extract dictionaries
    characters = [
        {
            "name": val["name"],
            "value": val["value"],
            "color": val["colour"],
            "index": index
        }
        for index, val in enumerate(data["nodes"])
    ]

    # extract interactions
    interactions = [(interaction["source"], interaction["target"], interaction["value"]) for interaction in data["links"]]

    return characters, interactions

def convert_to_2D_array_1(interactions, characters):
    """
    Convert the list of triples to a 2D NumPy array
    IN: interactions, list of tuple, interactions
        characters, list of dict, characters
    OUT: ndarray of shape (n, n), 2D array of interactions
    """
    n = len(characters)
    M = np.zeros((n, n), dtype = int)

    for source, target, value in interactions:
        M[source, target] += value
        M[target, source] += value
    
    for character in characters:
        index = character["index"]
        M[index, index] = character["value"]
    
    return M
def convert_to_unweighted_1(M):
    """
    Convert the 2D NumPy array to an unweighted 2D NumPy array
    IN: M, ndarray of shape (n, n), 2D array of interactions
    OUT: ndarray of shape (n, n), unweighted 2D array of interactions
    """
    M_unweighted = (M > 0).astype(int) # 1 if M[i, j] > 0, 0 otherwise
    np.fill_diagonal(M_unweighted, 0) # diagonal is 0

    return M_unweighted
def normalize_array_1(M_unweighted):
    """
    Normalize the 2D NumPy array
    IN: M_unweighted, ndarray of shape (n, n), unweighted 2D array of interactions
    OUT: ndarray of shape (n, n), normalized 2D array of interactions
    """
    # calculate sum of each column
    column_sums = M_unweighted.sum(axis = 0)

    # avoid division by 0
    column_sums[column_sums == 0] = 1

    M_normalized = M_unweighted / column_sums

    return M_normalized
def initialize_pagerank_1(n):
    """
    Initialize the PageRank vector
    IN: n, int, number of characters
    OUT: ndarray of shape (n, 1), PageRank vector
    """
    return np.full((n, 1), 1 / n)
def update_pagerank_1(r, M_normalized, alpha=0.85):
    """
    Update the PageRank vector
    IN: r, ndarray of shape (n, 1), PageRank vector
        M_normalized, ndarray of shape (n, n), normalized 2D array of interactions
        alpha, float, damping factor
    OUT: ndarray of shape (n, 1), updated PageRank vector
    """
    n = len(r)
    new_r = (alpha * np.matmul(M_normalized, r)) + ((1 - alpha) * (np.ones((n, 1)) / n))

    return new_r
def get_difference_1(r1, r2):
    """
    Calculate the difference between two PageRank vectors
    IN: r1, ndarray of shape (n, 1), PageRank vector
        r2, ndarray of shape (n, 1), PageRank vector
    OUT: float, difference between two PageRank vectors
    """
    distance = np.linalg.norm(r1 - r2)
    
    return distance
def get_pagerank_1(M_normalized, alpha=0.85, threshold=1e-6, max_iter=1000):
    """
    Calculate the PageRank vector
    IN: M_normalized, ndarray of shape (n, n), normalized 2D array of interactions
        alpha, float, damping factor
        threshold, float, threshold for the difference between the current and previous PageRank vectors
        max_iter, int, maximum number of iterations
    OUT: ndarray of shape (n, 1), PageRank vector
    """
    # initialize PageRank vector
    n = M_normalized.shape[0]
    curr = initialize_pagerank_1(n)

    # initialize previous vector
    prev = np.zeros_like(curr)

    for i in range(max_iter):
        curr = update_pagerank_1(curr, M_normalized, alpha) # update PageRank vector
        difference = get_difference_1(curr, prev)

        # if difference is less than specified threshold, break and return current vector
        if difference < threshold:
            break

        prev = curr.copy()
    
    return curr
def get_top_nodes_1(r, characters, k=1):
    """
    Return the top k nodes with the highest PageRank values
    IN: r, ndarray of shape (n, 1), PageRank vector
        characters, list of dict, characters
        k, int, number of top nodes
    OUT: list of dict, top k nodes
    """
    # flatten to 1D array
    r_flat = r.flatten()

    # get top k indices with highest values (descending order)
    top_indices = np.argsort(r_flat)[-k:][::-1]

    # get top k nodes using top k indices
    top_k_nodes = [characters[i] for i in top_indices]

    return top_k_nodes
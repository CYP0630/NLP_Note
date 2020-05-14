import numpy as np
from utils.utils import normalizeRows

def cos_sim(a, b):
	"""Takes 2 vectors a, b and returns the cosine similarity according
	to the definition of the dot product
	"""
	dot_product = np.dot(a, b)
	norm_a = np.linalg.norm(a)
	norm_b = np.linalg.norm(b)

	return dot_product / (norm_a * norm_b)

def knn(vector, matrix, k=10):
    nearest_idx = []

    ### YOUR CODE
    score = []
    for index, row in enumerate(matrix):
        score.append((cos_sim(row, vector), index))
    ### END YOUR CODE
    sorted_vectors = sorted(score, key=lambda x:x[0], reverse=True)
    for i in range(k):
        nearest_idx.append(sorted_vectors[i][1])

    return nearest_idx


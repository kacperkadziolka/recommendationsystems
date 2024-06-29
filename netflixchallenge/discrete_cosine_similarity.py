import numpy as np
import os
import math
from scipy import sparse
from scipy.fftpack import dct


def calculate_discrete_cosine_similarity(pair, data_matrix):
    col1 = data_matrix[:, pair[0]]
    col2 = data_matrix[:, pair[1]]

    # Apply Discrete Cosine Transform (DCT)
    dct_col1 = dct(col1.toarray().flatten(), type=2)
    dct_col2 = dct(col2.toarray().flatten(), type=2)

    norm_col1 = np.linalg.norm(dct_col1)
    norm_col2 = np.linalg.norm(dct_col2)

    if np.linalg.norm(dct_col1) == 0 or np.linalg.norm(dct_col2) == 0:
        return 0.0

    # Calculate discrete cosine similarity
    discrete_cos_sim = np.dot(dct_col1, dct_col2) / (norm_col1 * norm_col2)

    # Convert discrete cosine similarity to distance
    theta = math.degrees(np.arccos(discrete_cos_sim))
    return 1 - (theta / 180)


def compute_bitwise_hash(data_matrix, hash_matrix):
    dot_products = data_matrix.T.dot(hash_matrix.T)
    hash_values = (dot_products > 0).astype(int)
    print(hash_values)
    return hash_values


class RandomProjectionDCS:
    def __init__(self, seed, data_path, b=8, r=25, h=100):
        self.output_path = None
        np.random.seed = seed
        self.data_path = data_path
        self.b = b
        self.r = r
        self.h = h

    '''
    Method to generate the hyperplanes
    Create a set of h hyperplanes, with data_matrix.shape[0] dimensions
    '''
    def generate_random_vectors(self, data_matrix):
        plane_norms = np.random.rand(self.h, data_matrix.shape[0]) - .5
        print("plane norms", plane_norms)
        return plane_norms

    def lsh(self, hash_matrix, data_matrix):
        pairs = []

        checked_pairs = set()

        for band in range(self.b):
            print(band)
            buckets = {}
            num_potential_pairs = 0
            pairs_checked = 0  # Set to keep track of already checked pairs

            for col in range(hash_matrix.shape[0]):
                band_signatures = [hash_matrix[col, row] for row in range(band * self.r, (band + 1) * self.r)]
                bucket_key = hash(tuple(band_signatures))
                # Store the column index in the corresponding bucket
                if bucket_key not in buckets:
                    buckets[bucket_key] = [col]
                else:
                    buckets[bucket_key].append(col)
            print(len(buckets.keys()))
            for bucket_key, candidate_col in buckets.items():
                # print(bucket_key, len(candidate_col))
                if len(candidate_col) >= 2:
                    potential_pairs = set()
                    for i in range(len(candidate_col)):
                        for j in range(i + 1, len(candidate_col)):
                            potential_pairs.add((candidate_col[i], candidate_col[j]))
                    pairs_to_check = potential_pairs - checked_pairs  # Exclude pairs that have already been checked
                    num_potential_pairs += len(potential_pairs)
                    pairs_checked += len(pairs_to_check)
                    for p in pairs_to_check:
                        discrete_cosine_similairty = calculate_discrete_cosine_similarity(p, data_matrix)
                        checked_pairs.add(p)  # Mark the pair as checked
                        if discrete_cosine_similairty > 0.73:
                            print('Pair found:', p, discrete_cosine_similairty)
                            pairs.append(p)
                            with open(self.output_path, 'a') as output_file:
                                output_file.write(f"{p[0]}, {p[1]}\n")
                                output_file.close()
            print('Potential pairs in band:', num_potential_pairs)
            print('Pairs checked in band:', pairs_checked)
            print('Pairs so far:', len(pairs))

        return pairs

    def create_directory_file(self, output_directory='output', output_file='dcs.txt'):
        os.makedirs(output_directory, exist_ok=True)
        self.output_path = os.path.join(output_directory, output_file)

        # create new file every each independent run
        with open(self.output_path, 'w'):  # Create an empty file
            pass

    def run(self):
        print("discrete cosine similarity is running")
        self.create_directory_file()
        data = np.load(self.data_path)

        data_matrix = sparse.csc_matrix((data[:, 2], (data[:, 1], data[:, 0])))
        hash_matrix = self.generate_random_vectors(data_matrix)
        hash_values = compute_bitwise_hash(data_matrix, hash_matrix)

        pairs_found = self.lsh(hash_values, data_matrix)
        print("Total pairs found:", len(pairs_found))

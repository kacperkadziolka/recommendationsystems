import numpy as np
import os
from tqdm import tqdm
from scipy import sparse


def calculate_jaccardsimilarity(pair, data):
    col1 = data.getcol(pair[0]).indices
    col2 = data.getcol(pair[1]).indices

    intersection = set(col1).intersection(set(col2))
    union = set(col1).union(set(col2))
    jaccard_sim = len(intersection) / len(union)
    return jaccard_sim


class MinLSH:
    def __init__(self, seed, data_path, b=11, r=7, h=80):
        self.output_path = None
        np.random.seed = seed
        self.data_path = data_path
        self.b = b
        self.r = r
        self.h = h

    def create_hash(self, size):
        # Create a 2D array where each row is a range of integers from 0 to 'size - 1'
        hashes = np.tile(np.arange(size), (self.h, 1))

        # Shuffle each row independently
        np.apply_along_axis(np.random.shuffle, axis=1, arr=hashes)

        return hashes

    def create_sig_matrix(self, data_matrix, hash_matrix):
        '''
        The hash values are computed for each column in the data matrix, and the minimum hash value for each hash function is stored in the signature matrix.
        '''
        sig_matrix = np.zeros((self.h, data_matrix.shape[1]))  # Initialize the signature matrix with zeros
        data_matrix_indices = data_matrix.indices
        data_matrix_indptr = data_matrix.indptr

        # Loop through the columns of the data_matrix
        for i in tqdm(range(data_matrix.shape[1])):
            col_indices = data_matrix_indices[data_matrix_indptr[i]:data_matrix_indptr[
                i + 1]]  # Get the indices of non-zero elements in the current column
            mask = np.ones((self.h, data_matrix.shape[0]), dtype=bool)  # Create a mask filled with ones
            mask[:, col_indices] = False
            masked_hash = np.where(mask, data_matrix.shape[0] + 1,
                                   hash_matrix)  # Generate a masked_hash matrix based on the mask and hash_matrix
            sig_matrix[:, i] = np.min(masked_hash, axis=1)
        return sig_matrix

    def lsh(self, signature_matrix, data_matrix):
        pairs = []

        checked_pairs = set()

        for band in range(self.b):
            buckets = {}
            num_potential_pairs = 0
            pairs_checked = 0  # Set to keep track of already checked pairs

            for col in range(signature_matrix.shape[1]):
                band_signatures = [signature_matrix[row, col] for row in range(band * self.r, (band + 1) * self.r)]
                bucket_key = hash(tuple(band_signatures))
                # Store the column index in the corresponding bucket
                if bucket_key not in buckets:
                    buckets[bucket_key] = [col]
                else:
                    buckets[bucket_key].append(col)
            print(len(buckets.keys()))
            for bucket_key, candidate_col in buckets.items():
                if len(candidate_col) >= 2:
                    potential_pairs = set()
                    for i in range(len(candidate_col)):
                        for j in range(i + 1, len(candidate_col)):
                            potential_pairs.add((candidate_col[i], candidate_col[j]))
                    pairs_to_check = potential_pairs - checked_pairs  # Exclude pairs that have already been checked
                    num_potential_pairs += len(potential_pairs)
                    pairs_checked += len(pairs_to_check)
                    for p in pairs_to_check:
                        jaccard_similarity = calculate_jaccardsimilarity(p, data_matrix)
                        checked_pairs.add(p)  # Mark the pair as checked
                        if jaccard_similarity > 0.5:
                            pairs.append(p)
                            with open(self.output_path, 'a') as output_file:
                                output_file.write(f"{p[0]}, {p[1]}\n")
                                output_file.close()

            print('Potential pairs in band:', num_potential_pairs)
            print('Pairs checked in band:', pairs_checked)
            print('Pairs so far:', len(pairs))

        return pairs

    def create_directory_file(self, output_directory='output', output_file='js.txt'):
        os.makedirs(output_directory, exist_ok=True)
        self.output_path = os.path.join(output_directory, output_file)

        # create new file every each independent run
        with open(self.output_path, 'w'):  # Create an empty file
            pass

    def run(self):
        print("jaccard similarity is running")
        self.create_directory_file()
        data = np.load(self.data_path)

        data_matrix = sparse.csc_matrix((data[:, 2], (data[:, 1], data[:, 0])))
        hash_matrix = self.create_hash(data_matrix.shape[0])
        sig = self.create_sig_matrix(data_matrix, hash_matrix)

        pairs_found = self.lsh(sig, data_matrix)
        print("Total pairs found:", len(pairs_found))

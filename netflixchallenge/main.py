import argparse
import time

from jaccard_similarity import MinLSH
from cosine_similarity import RandomProjectionCS
from discrete_cosine_similarity import RandomProjectionDCS

argparser = argparse.ArgumentParser()
argparser.add_argument("-d", default='data/user_movie_rating.npy', type=str, help="Data file path")
argparser.add_argument("-s", default=42, type=int, help="Random seed")
argparser.add_argument("-m", default='js', type=str, help="Similarity measure (js / cs / dcs)")
args = argparser.parse_args()

if args.m == 'js':
    similarity_measure = MinLSH
if args.m == 'cs':
    similarity_measure = RandomProjectionCS
if args.m == 'dcs':
    similarity_measure = RandomProjectionDCS

start_time = time.time()

similarity_measure(args.s, args.d).run()

end_time = time.time()
elapsed_time = end_time - start_time
elapsed_minutes = elapsed_time / 60
print(f"Elapsed time: {elapsed_minutes:.2f} minutes")

import numpy as np
from sklearn.preprocessing import normalize
from generate import GENERATE
import random
import codecs


vocab = codecs.open("brown_vocab_100.txt")

#load the indices dictionary
word_index_dict = {k.rstrip(): v for v, k in enumerate(vocab.read().splitlines())}
counts = np.zeros((len(word_index_dict), len(word_index_dict))) + 0.1

f = codecs.open("brown_100.txt")
for l in f.read().splitlines():
    words = [i.lower() for i in l.split()]
    indices = np.apply_along_axis(lambda x: np.array([word_index_dict[i] for i in x]), 1, 
                                  np.lib.stride_tricks.sliding_window_view(np.array(words), 2))
    # print(indices)
    np.add.at(counts, (indices[:, 0], indices[:, 1]), 1)

probs = normalize(counts, norm='l1', axis=1)
spec_ind = np.array([
    [word_index_dict["all"], word_index_dict["the"]],
    [word_index_dict["the"], word_index_dict["jury"]],
    [word_index_dict["the"], word_index_dict["campaign"]],
    [word_index_dict["anonymous"], word_index_dict["calls"]],
])

spec_probs = probs[spec_ind[:, 0], spec_ind[:, 1]]

np.savetxt('smooth_probs.txt', spec_probs, fmt='%.10f')

f.close()
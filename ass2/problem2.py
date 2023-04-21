#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from generate import GENERATE


#load the indices dictionary
word_index_dict = {k.rstrip(): v for v, k in enumerate(open('brown_vocab_100.txt', encoding='utf-8').read().splitlines())}

counts = np.zeros((len(word_index_dict)))
with open('brown_100.txt', encoding='utf-8') as f:
    for line in f:
        counts += np.bincount([word_index_dict[_.lower()] for _ in line.split()], minlength=len(word_index_dict))

#TODO: normalize and writeout counts. 
probs = counts/np.sum(counts)
np.savetxt('unigram_probs.txt', probs, fmt='%.10f')

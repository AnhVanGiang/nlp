import numpy as np
from sklearn.preprocessing import normalize
from generate import GENERATE
import random
import codecs


vocab = codecs.open("brown_vocab_100.txt")

#load the indices dictionary
word_index_dict = {k.rstrip(): v for v, k in enumerate(vocab.read().splitlines())}
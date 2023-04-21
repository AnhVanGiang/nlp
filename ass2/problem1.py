#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""



# TODO: read brown_vocab_100.txt into word_index_dict
word_index_dict = {k.rstrip(): v for v, k in enumerate(open('brown_vocab_100.txt', encoding='utf-8').read().splitlines())}
# TODO: write word_index_dict to word_to_index_100.txt
with open('word_to_index_100.txt', 'w', encoding='utf-8') as f:
    for k, v in word_index_dict.items():
        f.write(f'{k}\t{v}\n')



print(word_index_dict['all'])
print(word_index_dict['resolution'])
print(len(word_index_dict))

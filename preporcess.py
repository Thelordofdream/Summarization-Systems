import numpy as np
import os


def cleanText(corpus):
    punctuation = ".\",?!:;(){}[]-_*"
    corpus = [z.lower().replace('\n', '') for z in corpus]
    corpus = [z.replace('\xc3\xa9', 'e') for z in corpus]
    corpus = [z.replace('\xc3\xae', 'i') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]

    # treat punctuation as individual words
    for c in punctuation:
        corpus = [z.replace(c, ' ') for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus


data_path = "aclImdb/train/"
pos_files = os.listdir(data_path + "pos/")
neg_files = os.listdir(data_path + "neg/")
pos_reviews = []
neg_reviews = []

for pos_file in pos_files[:5]:
    with open(data_path + "pos/" + pos_file, 'r') as infile:
        pos_reviews.extend(infile.readlines())
for neg_file in neg_files[:5]:
    with open(data_path + "neg/" + neg_file, 'r') as infile:
        neg_reviews.extend(infile.readlines())

pos_reviews = cleanText(pos_reviews)
neg_reviews = cleanText(neg_reviews)

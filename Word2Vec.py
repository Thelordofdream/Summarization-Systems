import numpy as np
import os
import gensim


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


def buildWordVector(model_w2v, text, size):
    sen = []
    vec = np.zeros(size).reshape((1, size))
    for word in text:
        try:
            vec = model_w2v[word].reshape((1, size))
            sen.extend(vec)
        except:
            continue
    return sen


def storeVecs(input, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(input, fw)
    fw.close()


def embedding(tag):
    data_path = "aclImdb/" + tag + "/"
    pos_files = os.listdir(data_path + "pos/")
    neg_files = os.listdir(data_path + "neg/")
    pos_reviews = []
    neg_reviews = []

    for pos_file in pos_files:
        with open(data_path + "pos/" + pos_file, 'r') as infile:
            pos_reviews.extend(infile.readlines())
    for neg_file in neg_files:
        with open(data_path + "neg/" + neg_file, 'r') as infile:
            neg_reviews.extend(infile.readlines())

    pos_reviews = cleanText(pos_reviews)
    neg_reviews = cleanText(neg_reviews)

    model_google = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    n_dim = 300
    pos_vector = [buildWordVector(model_google, z, n_dim) for z in pos_reviews]
    neg_vector = [buildWordVector(model_google, z, n_dim) for z in neg_reviews]
    storeVecs(pos_vector, tag + '/pos_vecs.pkl')
    storeVecs(neg_vector, tag + '/neg_vecs.pkl')
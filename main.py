import Word2Vec
import numpy as np
import random

# Word2Vec.embedding('train')
# Word2Vec.embedding('test')

train_pos_vector = Word2Vec.grabVecs("train/pos_vecs.pkl")
train_neg_vector = Word2Vec.grabVecs("train/neg_vecs.pkl")
test_pos_vector = Word2Vec.grabVecs("test/pos_vecs.pkl")
test_neg_vector = Word2Vec.grabVecs("test/neg_vecs.pkl")

m_max = 300
m_min = 0

train_pos = 0
train_pos_vector_alignment = []
for a in train_pos_vector:
    m = len(a)
    if m_max >= m >= m_min:
        train_pos += 1
        sentence = np.zeros((m_max, 300))
        start = int((m_max - m) / 2)
        sentence[start:start + m] = a
        train_pos_vector_alignment.append(sentence)
# Word2Vec.storeVecs(train_pos_vector_alignment, 'data/train/pos_vecs.pkl')

train_neg = 0
train_neg_vector_alignment = []
for a in train_neg_vector:
    m = len(a)
    if m_max >= m >= m_min:
        train_neg += 1
        sentence = np.zeros((m_max, 300))
        start = int((m_max - m) / 2)
        sentence[start:start + m] = a
        train_neg_vector_alignment.append(sentence)
# Word2Vec.storeVecs(train_neg_vector_alignment, 'data/train/neg_vecs.pkl')

test_pos = 0
test_pos_vector_alignment = []
for a in test_pos_vector:
    m = len(a)
    if m_max >= m >= m_min:
        test_pos += 1
        sentence = np.zeros((m_max, 300))
        start = int((m_max - m) / 2)
        sentence[start:start + m] = a
        test_pos_vector_alignment.append(sentence)
# Word2Vec.storeVecs(test_pos_vector_alignment, 'data/test/pos_vecs.pkl')


test_neg = 0
test_neg_vector_alignment = []
for a in test_neg_vector:
    m = len(a)
    if m_max >= m >= m_min:
        test_neg += 1
        sentence = np.zeros((m_max, 300))
        start = int((m_max - m) / 2)
        sentence[start:start + m] = a
        test_neg_vector_alignment.append(sentence)
# Word2Vec.storeVecs(test_neg_vector_alignment, 'data/test/neg_vecs.pkl')

train = range(0, train_pos + train_neg)
random.shuffle(train)
train_data = []
train_label = []
for a in train:
    if a < train_pos:
        train_data.append(train_pos_vector_alignment[a])
        label = [0, 1]
        train_label.append(label)
    else:
        train_data.append(train_neg_vector_alignment[a - train_pos])
        label = [1, 0]
        train_label.append(label)
Word2Vec.storeVecs(train_data, 'data/train/vecs.pkl')
Word2Vec.storeVecs(train_label, 'data/train/label.pkl')

test = range(0, test_pos + test_neg)
random.shuffle(test)
test_data = []
test_label = []
for a in test:
    if a < test_pos:
        test_data.append(test_pos_vector_alignment[a])
        label = [0, 1]
        test_label.append(label)
    else:
        test_data.append(test_neg_vector_alignment[a - test_pos])
        label = [1, 0]
        test_label.append(label)
Word2Vec.storeVecs(test_data, 'data/test/vecs.pkl')
Word2Vec.storeVecs(test_label, 'data/test/label.pkl')

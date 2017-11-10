# coding=utf-8
import network
import tensorflow as tf
import matplotlib.pyplot as plt
import gensim
import Word2Vec
import numpy as np


def predict(model, data, label, sess, list, sentences):
    test_data = data
    test_q = data
    test_a = data
    test_label = np.array(label)
    test_data = test_data.reshape((-1, model.steps, model.inputs))
    test_q = test_q.reshape((-1, model.steps, model.inputs))
    test_a = test_a.reshape((-1, model.steps, model.inputs))
    res, label, weight = sess.run([tf.sigmoid(model.output, name=None), model.y, model.s],
                           feed_dict={model.x: test_data, model.q: test_q, model.a: test_a, model.y: test_label,
                                      model.keep_prob_d: 1.0, model.keep_prob_q: 1.0, model.keep_prob_a: 1.0})
    print res[0]
    print label[0]
    weight = [i for i in sess.run(tf.transpose(weight[0], [1, 0]))[0]]
    for i in range(len(list)):
        if not list[i] == -1:
            list[i] = weight[list[i]]
        else:
            list[i] = 0
    entropy = []
    for i in range(len(sentences)-1):
        sum = 0
        for x in range(sentences[i], sentences[i+1]):
            sum += list[x]
        entropy.append(sum)
    # plt.bar(range(len(weight)), weight)
    # plt.show()
    return entropy, list





def buildWordVector(model_w2v, text, size):
    sen = []
    vec = np.zeros(size).reshape((1, size))
    list = []
    for word in text:
        try:
            vec = model_w2v[word].reshape((1, size))
            sen.extend(vec)
            list.append(1)
        except:
            list.append(0)
            continue
    return sen, list


if __name__ == "__main__":
    # data = network.data(path="data/train/", index=20)
    reviews = []
    with open("aclImdb/train/neg/2911_3.txt", 'r') as infile:
        reviews.extend(infile.readlines())
        label = [[0, 1]]
    text = Word2Vec.cleanText(reviews)
    print reviews[0]
    sent = reviews[0].replace('...', '.').split('.')
    sentences = [len(Word2Vec.cleanText([i])[0]) for i in sent]
    sum = 0
    for i in range(len(sentences)):
        sum += sentences[i]
        sentences[i] = sum
    sentences.insert(0, 0)
    print sentences
    print "Loading models......"
    model_google = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',
                                                                   binary=True)
    print "Loading Google Model Finished."
    n_dim = 300
    sen, list = buildWordVector(model_google, text[0], n_dim)

    m_max = 100
    m_min = 50
    data = []
    m = len(sen)
    print m
    if m <= 100:
        sentence = np.zeros((m_max, 300))
        start = int((m_max - m) / 2)
        sentence[start:start + m] = sen
        data.append(sentence)
        data = np.array(data)
        count = 0
        for i in range(len(list)):
            if list[i] == 1:
                list[i] = count+start
                count += 1
            else:
                list[i] = -1
        my_network = network.Attentive_Reader(name="TC")
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, "model/model.ckpt")
            entropy, list = predict(my_network, data, label, sess, list, sentences)
            print text[0][list.index(max(list))]
            print entropy
            print sent[entropy.index(max(entropy))] + "."
            plt.bar(range(len(entropy)), entropy)
            plt.show()

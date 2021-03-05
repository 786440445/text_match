import os
import pickle
import numpy as np

home_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

embedding_pickle_file = os.path.join(home_dir, 'embedding_table.pkl')


def load_embedding_txt():
    embeddings_index = {}
    with open(os.path.join(home_dir, 'tx_word2vec/500000-small.txt'), 'r') as f:
        for i,line in enumerate(f):
            if i == 0:
                continue
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


def init_embedding_table(embeddings_index, word_index, embedding_pickle_file, EMBEDDING_DIM=200):
    embedding_matrix = np.zeros((len(word_index), EMBEDDING_DIM), dtype='float32')
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    with open(os.path.join(home_dir, embedding_pickle_file), 'wb') as f:
        pickle.dump(embedding_matrix, f)


def get_word_index():
    word_index = {}
    with open(os.path.join(home_dir, 'data/vocab_words.txt'), 'r', encoding='utf-8') as f:
        for index, line in enumerate(f.readlines()):
            line = line.strip().split()
            word_index[line[0]] = index
    return word_index


def load_embedding_table():
    with open(os.path.join(home_dir, embedding_pickle_file), 'rb') as f:
        embedding_table = pickle.load(f)
    return embedding_table



if __name__ == '__main__':
    word_index = get_word_index()
    embeddings_index = load_embedding_txt()
    init_embedding_table(embeddings_index, word_index, embedding_pickle_file, EMBEDDING_DIM=200)
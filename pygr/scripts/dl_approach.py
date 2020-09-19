import os

import pandas as pd
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from tqdm import tqdm

from pygr.nlp import utils as utils

data_path = '/home/yus/data/real_or_not/'
train_path = os.path.join(data_path, 'train.csv')
test_path = os.path.join(data_path, 'test.csv')
submission_path = os.path.join(data_path, 'sample_submission.csv')

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Let's take a look at the dimentions of the data
print("CORPUS")
print("Train dims: {}".format(train.shape))
print("Test dims: {}".format(test.shape))

# Data cleaning, the order matters, removing punct first makes correctors not work properly
print("\nDATA CLEANING")
joined_corpus = ' '.join(train["text"].values)
print("Characters before cleaning : {}".format(len(joined_corpus)))

removed_urls = train["text"].apply(lambda x: utils.find_url(x)).dropna()
train["text"] = train["text"].apply(lambda x: utils.remove_html(x))
print("Characters after HMTL cleaning : {}".format(len(' '.join(train["text"].values))))

removed_html = train["text"].apply(lambda x: utils.find_html(x)).dropna()
train["text"] = train["text"].apply(lambda x: utils.remove_url(x))
print("Characters after URL cleaning : {}".format(len(' '.join(train["text"].values))))

removed_punct = train["text"].apply(lambda x: utils.find_punct(x))
train["text"] = train["text"].apply(lambda x: utils.remove_punct(x)).dropna()
print("Characters after punct cleaning : {}".format(len(' '.join(train["text"].values))))

# To lowease
train["text"] = train["text"].apply(lambda x: x.lower())

# Look for contractions in english and substitute them
train["text"] = train["text"].apply(lambda x: utils.expand_contraction(x))

glove_path = "/home/yus/data/glove/glove.6B/glove.6B.100d.txt"
glove_w2v_format_path = glove_path.replace(".txt", ".w2v_format.txt")
print("\nLoading GloVe word vectors...")
vectors = KeyedVectors.load_word2vec_format(glove_w2v_format_path)
print("Done. {} loaded.".format(glove_path.split('/')[-1]))

if not os.path.exists(glove_w2v_format_path):
    _ = glove2word2vec(glove_input_file=glove_path, word2vec_output_file=glove_w2v_format_path)

corpus = utils.create_corpus(dataframe=train, text_column='text', as_sentences=False, only_alpha=True)
vocabulary = utils.build_vocab(corpus)
glove_vocabulary = list(vectors.vocab.keys())
oov = utils.check_embedding_coverage(vocabulary=vocabulary, keyed_vectors=vectors)
non_alpha = [term for term in oov if not term[0].isalpha()]


# MAX_LEN = 50
# tokenizer_obj = Tokenizer()
# tokenizer_obj.fit_on_texts(corpus)
# sequences = tokenizer_obj.texts_to_sequences(corpus)
#
# train_pad = pad_sequences(sequences, maxlen=MAX_LEN, truncating='post', padding='post')
#
# word_index = tokenizer_obj.word_index
# print("Number of unique words: {}".format(len(word_index)))
#
# n_words = len(word_index) + 1
# embedding_dims = 100
# embedding_matrix = np.zeros((n_words, embedding_dims))
#
# index_to_word = dict()
# for word, i in tqdm(word_index.items()):
#     if i > n_words:
#         continue
#     index_to_word[i] = word
#     print(word, i)
#     vector = word_vectors.get(word)
#     if vector is not None:
#         embedding_matrix[i] = vector
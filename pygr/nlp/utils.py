"""This file contains functions used to preprocess text for Natural Language Processing tasks."""
import os
import re
import operator
import string
from typing import List, Dict

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from tqdm import tqdm

WordList = List[str]

URL_REGEX = r'https?://\S+|www\.\S+'
HTML_REGEX = r'<.*?>'
EMOJIS_REGEX = "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-" \
               "\U000027B0\U000024C2-\U0001F251]+"


def remove_url(text: str) -> str:
    """Remove the URLs from the text.

    :param text: Text to be cleaned.
    :return: Clean text.
    """
    url = re.compile(URL_REGEX)
    return url.sub(r'', text)


def find_url(text: str) -> List[str]:
    """Find the URLs in text.

    :param text: Text to be analyzed.
    :return: URL if any.
    """
    out = re.findall(pattern=URL_REGEX, string=text)

    if out:
        return out


def remove_html(text: str) -> str:
    """Remove the HTML tags from the text.

        :param text: Text to be cleaned.
        :return: Clean text.
        """
    return re.sub(pattern=HTML_REGEX, repl='', string=text)


def find_html(text: str) -> List[str]:
    """Find the HTML tags in text.

    :param text: Text to be analyzed.
    :return: List of HTML tags if any.
    """
    out = re.findall(pattern=HTML_REGEX, string=text)

    if out:
        return out


def remove_emojis(text: str) -> str:
    """Remove the emojis from the text.

    :param text: Text to be cleaned.
    :return: Clean text.
    """
    return re.sub(pattern=EMOJIS_REGEX, repl='', string=text, flags=re.UNICODE)


def remove_punct(text: str) -> str:
    """Remove punctuation from the text.

    :param text: Text to be cleaned.
    :return: Clean text.
    """
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)


def find_punct(text: str) -> List[str]:
    """Find the HTML tags in text.

    :param text: Text to be analyzed.
    :return: List of HTML tags if any.
    """
    punct = string.punctuation
    punct_found = []
    for p in punct:
        if p in text:
            punct_found.append(p)

    if punct_found:
        return punct_found


def create_corpus(
        dataframe: pd.DataFrame, text_column: str, as_sentences: bool = True, only_alpha: bool = False
) -> List:
    """Create a corpus from text data.

    :param dataframe: Dataframe containing the text data.
    :param text_column: String with the name of the column with the text data.
    :param as_sentences: If True the documents will be sentences (string) instead of a list of words.
    :param only_alpha: It True, only keep words with alphabetical characters. This should remove a lot of words.
    :return: Corpus of sentences or lists of words.
    """
    corpus = list()

    if text_column not in list(dataframe.columns):
        raise ValueError("Incorrect name for the text column. Not in {}.".format(dataframe.columns))

    text = dataframe[text_column]

    for document in tqdm(text, desc="Creating corpus"):
        if only_alpha:
            words = [word.lower() for word in word_tokenize(document) if word.isalpha()]
        else:
            words = [word.lower() for word in word_tokenize(document)]

        corpus.append(words)

    if as_sentences:
        corpus = [' '.join(words) for words in corpus]

    return corpus


def load_glove(path: str) -> Dict:
    """This functions load the word vectors stored in the given path into a dictionary.

    :param path: Path to the glove .txt
    :return: Dictionary with the vectors where the keys are the words.
    """
    word_vectors = dict()

    with open(path, "r") as file_handler:
        for line in file_handler:
            values = line.split()
            word = values[0]
            vectors = np.array(values[1:], 'float32')
            word_vectors[word] = vectors

    return word_vectors


def tokenized_df(corpus: List[List[str]], labels: List[str]) -> pd.DataFrame:
    """Given a tokenized corpus and a list of labels, create a dataframe that looks like this:

     	                                            texts   label
    -------------------------------------------------------------
    0 	[asian, exporters, fear, damage, japan, rift, ... 	trade
    1 	[china, daily, vermin, eat, pct, grain, stocks... 	grain
    2 	[australian, foreign, ship, ban, ends, nsw, po... 	ship
    3 	[sumitomo, bank, aims, quick, recovery, merger... 	acq
    4 	[amatil, proposes, two, for, bonus, share, iss... 	earn

    :param corpus: List of tokenized words. List of lists containing strings.
    :param labels: Labels for a classification problem.
    :return: pandas.Dataframe with two columns, one named 'text' with the list of words and 'label'.
    """
    n_documents = len(corpus)
    n_labels = len(labels)

    if n_documents != n_labels:
        raise ValueError("The number of documents and labels do not match.")

    df = pd.DataFrame(data=None, columns=['text', 'label'])

    for i in tqdm(range(0, n_documents), desc="Creating tokenized df"):
        df = df.append({'text': corpus[i], 'label': labels[i]}, ignore_index=True)

    return df


def vectorized_embeddings(words: List[str], word_vectors: Dict[str, np.array]):
    """For a sequence of words, average the word vectors to have a document level vector representation of the text.

    :param words: List of words, the document.
    :param word_vectors: Word vectors in a dictionary where the key is the word and the value is the word vector.
    :return:
    """
    vocab = list(word_vectors.keys())
    dims = list(word_vectors.values())[0].shape[0]

    valid_words = [word for word in words if word in vocab]

    if valid_words:
        embedding = np.zeros((len(valid_words), dims), dtype=np.float32)

        for idx, word in enumerate(valid_words):
            embedding[idx] = word_vectors[word]

        return np.mean(embedding, axis=0)

    else:
        return np.zeros(dims)


def build_vocab(sentences: List[str]) -> Dict[str, int]:
    """Build a vocabulary using a list of sentences.

    :param sentences: List of lists of words.
    :return: Dictionary with the frequency of each word in the vocabulary.
    """
    vocab = {}

    for sentence in tqdm(sentences, desc="Sentences processed"):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab


def check_embedding_coverage(vocabulary: Dict[str, int], keyed_vectors: KeyedVectors):
    """See what words from the vocabulary are not represented in the word vectors.

    Output information about the OOV (out of vocabulary) terms.

    :param vocabulary: Dictionary with words as keys and frequencies of the words in the corpus as values.
    :param keyed_vectors: gensim.model.KeyedVectors instance containing the word vectors.
    :return:
    """
    cov = {}  # Covered words dictionary
    oov = {}  # Out of vocabulary dictionary
    covered_words = 0
    oov_words = 0

    for word in tqdm(vocabulary, desc="Words checked"):
        try:
            vector = keyed_vectors.get_vector(word)
            cov[word] = vector
            covered_words += vocabulary[word] if isinstance(vocabulary, Dict) else word

        except KeyError:
            oov[word] = vocabulary[word] if isinstance(vocabulary, Dict) else word
            oov_words += vocabulary[word] if isinstance(vocabulary, Dict) else word

    found_vocab_vectors = len(cov) / len(vocabulary)
    found_vocab_all_text = covered_words / (covered_words + oov_words)
    print('Found embeddings for {:.2%} of vocab'.format(found_vocab_vectors))
    print('Found embeddings for  {:.2%} of all text'.format(found_vocab_all_text))
    sorted_oov = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_oov


english_contractions = {
    "he's": "he is",
    "there's": "there is",
    "we're": "we are",
    "that's": "that is",
    "won't": "will not",
    "they're": "they are",
    "can't": "cannot",
    "wasn't": "was not",
    "don\x89ûªt": "do not",
    "aren't": "are not",
    "isn't": "is not",
    "what's": "what is",
    "haven't": "have not",
    "hasn't": "has not",
    "there's": "there is",
    "he's": "he is",
    "it's": "it is",
    "you're": "you are",
    "i'm": "i am",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "i'm": "i am",
    "i\x89ûªm": "i am",
    "i'm": "i am",
    "isn't": "is not",
    "here's": "here is",
    "you've": "you have",
    "you\x89ûªve": "you have",
    "we're": "we are",
    "what's": "what is",
    "couldn't": "could not",
    "we've": "we have",
    "it\x89ûªs": "it is",
    "doesn\x89ûªt": "does not",
    "it\x89ûªs": "it is",
    "here\x89ûªs": "here is",
    "who's": "who is",
    "i\x89ûªve": "i have",
    "y'all": "you all",
    "can\x89ûªt": "cannot",
    "would've": "would have",
    "it'll": "it will",
    "we'll": "we will",
    "wouldn\x89ûªt": "would not",
    "we've": "we have",
    "he'll": "he will",
    "y'all": "you all",
    "weren't": "were not",
    "didn't": "did not",
    "they'll": "they will",
    "they'd": "they would",
    "don't": "do not",
    "that\x89ûªs": "that is",
    "they've": "they have",
    "i'd": "i would",
    "should've": "should have",
    "you\x89ûªre": "you are",
    "where's": "where is",
    "don\x89ûªt": "do not",
    "we'd": "we would",
    "i'll": "i will",
    "weren't": "were not",
    "they're": "they are",
    "can\x89ûªt": "cannot",
    "you\x89ûªll": "you will",
    "i\x89ûªd": "i would",
    "let's": "let us",
    "it's": "it is",
    "can't": "cannot",
    "don't": "do not",
    "you're": "you are",
    "i've": "i have",
    "that's": "that is",
    "i'll": "i will",
    "doesn't": "does not",
    "i'd": "i would",
    "didn't": "did not",
    "ain't": "am not",
    "you'll": "you will",
    "i've": "i have",
    "don't": "do not",
    "i'll": "i will",
    "i'd": "i would",
    "let's": "let us",
    "you'd": "you would",
    "it's": "it is",
    "ain't": "am not",
    "haven't": "have not",
    "could've": "could have",
    "youve": "you have"
}


def expand_contraction(text: str):
    """Search for contractions in the text and replace with the expanded version.

    :param text: Text to be updated with expanded contractions.
    :return: Update text if any contraction found.
    """
    for contraction in english_contractions.keys():
        expanded = english_contractions[contraction]
        update = re.sub(contraction, expanded, text)

        if text != update:
            print("Text updated: {} -> {}".format(contraction, expanded))
            text = update

    return text

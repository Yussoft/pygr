"""This main is the entry point used to train models using the standard text classification approach."""
import os
from tqdm import tqdm

import mlflow
import pandas as pd
import numpy as np

# Algorithms
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Model selection
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from pygr.mlflow import get_experiment_id
import pygr.nlp as utils

SEED = 13092020

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

# What are the columns???
print("\nDF Columns: {}".format(train.columns.to_list()))

# Data cleaning, the order matters, removing punct first makes correctors not work properly
print("\nDATA CLEANING")
joined_corpus = ' '.join(train["text"].values)
print("Characters before cleaning : {}".format(len(joined_corpus)))

train["text"] = train["text"].apply(lambda x: nlp_utils.remove_html(x))
print("Characters after HMTL cleaning : {}".format(len(' '.join(train["text"].values))))

train["text"] = train["text"].apply(lambda x: nlp_utils.remove_url(x))
print("Characters after URL cleaning : {}".format(len(' '.join(train["text"].values))))

train["text"] = train["text"].apply(lambda x: nlp_utils.remove_punct(x))
print("Characters after punct cleaning : {}".format(len(' '.join(train["text"].values))))


# Load glove vectors
embeddings_path = "/home/yus/data/glove/glove.6B/glove.6B.100d.txt"
print("\nLoading GloVe word vectors...")
word_vectors = nlp_utils.load_glove(path=embeddings_path)

# Create a corpus, tokenize each document and store it in a dataframe
corpus = nlp_utils.create_corpus(dataframe=train, text_column='text')
labels = train['target']
tokenized_df = nlp_utils.tokenized_df(corpus=corpus, labels=labels)

n_samples = tokenized_df.shape[0]
embeddings_dims = list(word_vectors.values())[0].shape[0]

# Create a feature vector using GloVe embeddings for each document.
features = np.zeros(shape=(n_samples, embeddings_dims))

for i in tqdm(range(0, n_samples), desc="Vectorizing embeddings"):
    words = tokenized_df['text'].iloc[i]
    features[i] = nlp_utils.vectorized_embeddings(words=words, word_vectors=word_vectors)

# Model definition and training
clf = GaussianNB()
# clf = SVC()
# clf = LGBMClassifier(n_jobs=8)
# clf = XGBClassifier(n_jobs=4)

# Randomized Search Cross Validation parameters
param_distributions = {
    # 'max_depth': [1, 2, 3],
    # 'eta': [0.01, 0.1, 1, 2],
    # 'objective': ['binary:logistic']
}
# Scoring list: https://scikit-learn.org/stable/modules/model_evaluation.html
scoring = {
    'accuracy': 'accuracy',
    'recall': 'recall',
    'precision': 'precision',
    'f1': 'f1'
}
n_iter = 10

random_search = RandomizedSearchCV(
    estimator=clf,
    param_distributions=param_distributions,
    n_iter=n_iter,
    scoring=scoring,
    n_jobs=3,
    refit='f1',
    cv=None,  # Default 5fold-CV
    random_state=SEED,
    return_train_score=True,
    verbose=1
)

result = random_search.fit(X=features, y=labels)
print("Best parameters: {}".format(random_search.best_params_))
print("Best score: {}".format(random_search.best_score_))
print("Best index: {}".format(random_search.best_index_))

scores = [
    'mean_train_accuracy', 'mean_test_accuracy', 'mean_train_recall', 'mean_test_recall', 'mean_train_precision',
    'mean_test_precision', 'mean_train_f1', 'mean_test_f1'
]
cv_results_df = pd.DataFrame(random_search.cv_results_)
best_metrics = cv_results_df.loc[random_search.best_index_, scores]

experiment_name = "ciklum_interview"
# run_name = "Baseline XGBoost model with GloVe 100D"
run_name = "Baseline GaussianNB model with GloVe 100D"
id = get_experiment_id(experiment_name)

with mlflow.start_run(run_id=None, experiment_id=id, run_name=run_name):
    # mlflow.log_param('Algorithm', "XGBoost Glove 100D")
    mlflow.log_param('Algorithm', "SVC GaussianNB 100D")

    for name, value in random_search.best_params_.items():
        mlflow.log_param(name, value)

    for name, value in best_metrics.items():
        mlflow.log_metric(name, value)

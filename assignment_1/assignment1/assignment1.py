# %%
from sklearn import linear_model
import sklearn
import json
from matplotlib import pyplot as plt
from collections import defaultdict
import numpy
import random
import gzip
import math
import string
import scipy
from scipy import sparse
from implicit import bpr
import tensorflow as tf
from collections import defaultdict

import gzip
from collections import defaultdict

from typing import List


# %%
def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r

# %%
def accuracy(pred, y):
    TP_ = numpy.logical_and(pred, y)
    FP_ = numpy.logical_and(pred, numpy.logical_not(y))
    TN_ = numpy.logical_and(numpy.logical_not(pred), numpy.logical_not(y))
    FN_ = numpy.logical_and(numpy.logical_not(pred), y)

    TP = sum(TP_)
    FP = sum(FP_)
    TN = sum(TN_)
    FN = sum(FN_)

    acc = (TP + TN)/len(pred)
    return acc


# %%
entire_dataset = []
booksPerUser_all = defaultdict(set)
usersPerBook_all = defaultdict(set)
ratingsPerBook_all = defaultdict(list)

# sparse interaction matrix

for l in readCSV("train_Interactions.csv.gz"):
    entire_dataset.append(l)

random.shuffle(entire_dataset)          # shuffle data to avoid overfitting

train_data = entire_dataset[:190000]

userIDs, itemIDs = {}, {}
for u, b, r in entire_dataset:
    booksPerUser_all[u].add(b)
    usersPerBook_all[b].add(u)
    ratingsPerBook_all[b].append(r)
    if not u in userIDs:
        userIDs[u] = len(userIDs)
    if not b in itemIDs:
        itemIDs[b] = len(itemIDs)

nUsers, nItems = len(userIDs), len(itemIDs)


# %% [markdown]
# ## Problem 1: Have read?
# 

# %%
# Build validation with 50% have read and 50% unread
valid_data = []
for u, b, _ in entire_dataset[190000:]:
    valid_data.append((u, b, 1))
notRead_valid_set = []
set_of_books = set([b for b in itemIDs])

booksPerUser_valid = defaultdict(set)
for u, b, r in valid_data:
    booksPerUser_valid[u].add(b)

for d in valid_data:
    # get the books that user have not read
    diff = set_of_books.difference(booksPerUser_valid[d[0]])
    notRead_valid_set.append(
        (d[0], list(diff)[random.randint(0, len(diff)-1)]))
    # notRead_valid_set[d[0]].append(list(diff)[random.randint(0, len(diff)-1)]) # get random book for user

# adding to current validation pairs of (u,b) of books that have not been read by user
valid_data_q1 = valid_data
for u, b in notRead_valid_set:
    valid_data_q1.append((u, b, 0))

random.shuffle(valid_data_q1)

items = list(itemIDs.keys())

# %%
# Initialize prediction data structure and test dataset
test_dataset = []
predictions = open("predictions_Read.csv", 'w')
with open("pairs_Read.csv") as test_data:
    for l in test_data:
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u, b = l.strip().split(',')
        test_dataset.append((u, b))
        # Check if user and books is in indexing data structure
        if u not in userIDs:
            userIDs[u] = len(userIDs)
        if b not in itemIDs:
            itemIDs[b] = len(itemIDs)


# %%
class BPRbatch(tf.keras.Model):
    def __init__(self, K, lamb):
        super(BPRbatch, self).__init__()
        # Initialize variables
        self.betaI = tf.Variable(
            tf.random.normal([len(itemIDs)], stddev=0.001))
        self.gammaU = tf.Variable(tf.random.normal(
            [len(userIDs), K], stddev=0.001))
        self.gammaI = tf.Variable(tf.random.normal(
            [len(itemIDs), K], stddev=0.001))
        # Regularization coefficient
        self.lamb = lamb

    # Prediction for a single instance
    def predict(self, u, i):
        p = self.betaI[i] + tf.tensordot(self.gammaU[u], self.gammaI[i], 1)
        return p

    # Regularizer
    def reg(self):
        return self.lamb * (tf.nn.l2_loss(self.betaI) +
                            tf.nn.l2_loss(self.gammaU) +
                            tf.nn.l2_loss(self.gammaI))

    def score(self, sampleU, sampleI):
        u = tf.convert_to_tensor(sampleU, dtype=tf.int32)
        i = tf.convert_to_tensor(sampleI, dtype=tf.int32)
        beta_i = tf.nn.embedding_lookup(self.betaI, i)
        gamma_u = tf.nn.embedding_lookup(self.gammaU, u)
        gamma_i = tf.nn.embedding_lookup(self.gammaI, i)
        x_ui = beta_i + tf.reduce_sum(tf.multiply(gamma_u, gamma_i), 1)
        return x_ui

    def call(self, sampleU, sampleI, sampleJ):
        x_ui = self.score(sampleU, sampleI)
        x_uj = self.score(sampleU, sampleJ)
        return -tf.reduce_mean(tf.math.log(tf.math.sigmoid(x_ui - x_uj)))


# %%
# Build model on new "userIDs" and "itemIDs" length because there were some users that DNE in the "train_interactions.csv.gz"
optimizer = tf.keras.optimizers.Adam(0.1)
modelBPR = BPRbatch(5, 0.00001)


# %%
def trainingStepBPR(model, interactions):
    Nsamples = 50000
    with tf.GradientTape() as tape:
        sampleU, sampleI, sampleJ = [], [], []
        for _ in range(Nsamples):
            u, i, _ = random.choice(interactions)  # positive sample
            j = random.choice(items)  # negative sample
            while j in booksPerUser_all[u]:
                j = random.choice(items)
            sampleU.append(userIDs[u])
            sampleI.append(itemIDs[i])
            sampleJ.append(itemIDs[j])

        loss = model(sampleU, sampleI, sampleJ)
        loss += model.reg()
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients((grad, var) for
                              (grad, var) in zip(
                                  gradients, model.trainable_variables)
                              if grad is not None)
    return loss.numpy()


# %%
for i in range(300):
    obj = trainingStepBPR(modelBPR, entire_dataset)
    if (i % 10 == 9): print("iteration " + str(i+1) + ", objective = " + str(obj))

# %%
itemsScorePerUser_test = defaultdict(list)
# Add prediction to prediction data structure
for u, b in test_dataset:
    pred = modelBPR.predict(userIDs[u], itemIDs[b]).numpy()
    itemsScorePerUser_test[u].append((pred, b))

# Sort prediction data structure by score
for u in itemsScorePerUser_test.keys():
    itemsScorePerUser_test[u].sort(reverse=True)


# %%
# Checking data in prediction data structure
for u in list(itemsScorePerUser_test.keys())[:10]:
    print(f"u: {u}, items: {itemsScorePerUser_test[u]}")


# %%
# Make prediction
y_pred_test = []
pred_data = []
read_cnt_test = 0
unread_cnt_test = 0
for u, b in test_dataset:
    len_before = len(y_pred_test)
    fst_half = len(itemsScorePerUser_test[u])//2
    if fst_half == 0 and read_cnt_test <= unread_cnt_test:
        y_pred_test.append(1)
        pred_data.append((u, b, 1))
        # predictions.write(u + ',' + b + ",1\n")
    elif fst_half == 0 and read_cnt_test > unread_cnt_test:
        y_pred_test.append(0)
        pred_data.append((u, b, 0))
        # predictions.write(u + ',' + b + ",0\n")
    else:
        for sb in itemsScorePerUser_test[u][:fst_half]:
            if b in sb:
                y_pred_test.append(1)
                pred_data.append((u, b, 1))
                # predictions.write(u + ',' + b + ",1\n")
                read_cnt_test += 1
                break
        if len_before == len(y_pred_test):
            y_pred_test.append(0)
            pred_data.append((u, b, 0))
            # predictions.write(u + ',' + b + ",0\n")
            unread_cnt_test += 1


# %%
# test that there is 50% read and 50% unread predictions
print(len(y_pred_test))
print(sum(y_pred_test))
print(pred_data[19999])

# %%
with open("predictions_Read.csv", 'w') as prediction_file:
    prediction_file.write("userID,bookID,prediction\n")
    for u, b, p in pred_data:
        prediction_file.write(u + ',' + b + ',' + str(p) + "\n")


# %% [markdown]
# ## Problem 2: Predict Category

# %%
# Gather all data
data = []

for d in readGz("train_Category.json.gz"):
    data.append(d)

print(data[0])

# Split training and vlaidation data
category_all_data = [d for d in data]
category_train_data = category_all_data[:90000]
category_valid_data = category_all_data[90000:]
# stemmer = PorterStemmer()
punct = string.punctuation

wordCount = defaultdict(int)
for d in category_all_data:
    rev: string = d['review_text']
    rev = rev.lower()                           # lowercase
    rev = [c for c in rev if not (c in punct)]  # remove punctuation (char)
    rev = ''.join(rev)
    words = rev.strip().split()
    for w in words:
        wordCount[w] += 1

# sort word by frequency
counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()

# get dictionary
words_dict = [x[1] for x in counts[:10000]]
words_dictID = dict(zip(words_dict, range(len(words_dict))))


# %%
# get document frequency using training data set
df = defaultdict(int)
for d in category_all_data:
    rev: string = d['review_text']
    rev = rev.lower()                           # lowercase
    rev = [c for c in rev if not (c in punct)]  # remove punctuation (char)
    rev = ''.join(rev)
    words = rev.strip().split()
    for w in set(words):
        df[w] += 1


# %%
# get review_text for each review and compute tf vector
def feature(data):
    tfidf_vector = [0]*len(words_dict)
    text = data['review_text']
    text = text.lower()                           # lowercase
    text = [c for c in text if not (c in punct)]  # remove punctuation (char)
    text = ''.join(text)
    words = text.strip().split()
    # build tfidf vector
    for w in words:
        if w in words_dict:
            tfidf_vector[words_dictID[w]] = (
                math.log2(len(category_all_data)/df[w]))
    return tfidf_vector


# %%
X = [feature(d) for d in category_all_data]
y = [d['genreID'] for d in category_all_data]


# %%
# train model
test_model = linear_model.LogisticRegression(C=0.001) # 0.743 with 0.001
test_model.fit(X,y)

# %%
reviewID_dict = {}
for d in readGz("test_Category.json.gz"):
    reviewID_dict[d['review_id']] = d


# %%
with open("predictions_Category.csv", 'w') as predictions:
    for l in open("pairs_Category.csv"):
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u, r = l.strip().split(',')
        x_test = [feature(reviewID_dict[r])]
        ypred = test_model.predict(x_test)
        predictions.write(u + ',' + r + "," + str(ypred[0]) + "\n")




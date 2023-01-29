# %%
import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model
import random

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

# %%
def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

# %%
def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r

# %%
answers = {}

# %%
# Some data structures that will be useful

# %%
allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)

# %%
len(allRatings)

# %%
allRatings[0]

# %%
ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))

# %%
predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"): # header
        predictions.write(l)
        continue
    u,b = l.strip().split(',') # Read the user and item from the "pairs" file and write out your prediction
    # (etc.)
    
predictions.close()

# %%
##################################################
# Read prediction                                #
##################################################

# %%
# Copied from baseline code
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalRead/2: break



# %%
### Question 1

# %%
booksPerUserValid = defaultdict(set)
usersPerBookAllRatings = defaultdict(set)

for u, b, r in allRatings:
    # get sets of users that have read book
    usersPerBookAllRatings[b].add(u)

# get set of books from ALL RATINGS
set_of_books = set([b for b in usersPerBookAllRatings])


for d in ratingsValid:
    # get sets of books for each user that they've read
    booksPerUserValid[d[0]].add(d[1])

print(f"books per user valid set: {[(u, books) for ind, (u, books) in enumerate(booksPerUserValid.items()) if ind < 2]}")
print(f"Users per book in all rating: {[(b, users) for ind, (b, users) in enumerate(usersPerBookAllRatings.items()) if ind < 2]}")
print(f"length of ratingValid: {len(ratingsValid)}")
print(f"booksPerUserValid length: {len(booksPerUserValid)}")

# %%
notRead_valid_set = []

for d in ratingsValid:
    # get the books that user have not read
    diff = set_of_books.difference(booksPerUserValid[d[0]])

    # add not_read books to existing validation set 
    notRead_valid_set.append((d[0], list(diff)[random.randint(0, len(diff)-1)]))
    # notRead_valid_set[d[0]].append(list(diff)[random.randint(0, len(diff)-1)]) # get random book for user


print(f"List (user, book) entries of unread books by user: {[(u, book) for ind, (u, book) in enumerate(notRead_valid_set) if ind < 2]}")
print(len(notRead_valid_set))

# %%
# adding to current validation set
print(f"before length: {len(ratingsValid)}")
ratingsValid_q1 = ratingsValid
for u, b in notRead_valid_set:
    ratingsValid_q1.append((u,b))

print(f"after length: {len(ratingsValid)}")

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

# Evaluate performance of baseline model on validation set
y_q1 = [1]*10000 + [0]*10000
ypred_q1 = []
count_exist = 0
for d in ratingsValid_q1:
    if d[1] in return1:
        ypred_q1.append(1)
    else:
        ypred_q1.append(0)

acc1 = accuracy(ypred_q1, y_q1)
print(acc1)

# %%
answers['Q1'] = acc1

# %%
assertFloat(answers['Q1'])

# %%
### Question 2

# %%
# reset rating valid and other data sets
# ratingsTrain = allRatings[:190000]
# ratingsValid = allRatings[190000:]
# ratingsPerUser = defaultdict(list)
# ratingsPerItem = defaultdict(list)
# for u, b, r in ratingsTrain:
#     ratingsPerUser[u].append((b, r))
#     ratingsPerItem[b].append((u, r))

# add 

acc2 = 0
threshold = 2
for thres in [0.05*r for r in range(1, 20)]:
    return2 = set()
    count_q2 = 0
    ypred_q2 = []
    for ic, i in mostPopular:
        count_q2 += ic
        return2.add(i)
        if count_q2 > totalRead*thres: break
    for d in ratingsValid:
        if d[1] in return2:
            ypred_q2.append(1)
        else:
            ypred_q2.append(0)
    curr_acc = accuracy(ypred_q2, y_q1)
    if curr_acc > acc2:
        acc2 = curr_acc
        threshold = thres

# %%
answers['Q2'] = [threshold, acc2]
print(threshold)
print(acc2)

# %%
assertFloat(answers['Q2'][0])
assertFloat(answers['Q2'][1])

# %% [markdown]
# ## Question 3

# %%
### Question 3/4

# %%
def Jaccard(s1, s2):
    numerator = len(s1.intersection(s2))
    denominator = len(s1.union(s2))
    if denominator == 0:
        return 0
    return numerator/denominator


# %%
# setting up data structures from training and validation data

booksPerUser_train = defaultdict(set)
booksPerUser_valid = defaultdict(set)
usersPerBook_train = defaultdict(set)
usersPerBook_valid = defaultdict(set)

for u, b, r in ratingsTrain:
    booksPerUser_train[u].add(b)
    usersPerBook_train[b].add(u)

for d in ratingsValid:
    booksPerUser_valid[d[0]].add(d[1])
    usersPerBook_valid[d[1]].add(d[0])


# %%
# for b in usersPerBook_valid:
#     sim = Jaccard(usersPerBook_train[u], usersPerBook_valid[u])
#     if sim > 0.5:
#         pass
acc3 = 0 
ypred_q3 = []
for d in ratingsValid:
    # get similaritiy for all pairs in validation set
    max_sim = 0                                                  # d[1] = b' = books from training set
    for b in booksPerUser_train[d[0]]:
        if b == d[1]: continue                                  # b = books from validation set, using the user in validation set
        sim = Jaccard(usersPerBook_train[b], usersPerBook_train[d[1]])  # compute jaccard similarity between users in train for books in validation set and training set
        if max_sim < sim:
            max_sim = sim
    if max_sim > 0:
        ypred_q3.append(1)
    else:
        ypred_q3.append(0)


# %%
acc3 = accuracy(ypred_q3, y_q1)
print(acc3)

# %% [markdown]
# ## Question 4

# %%
count_q4 = 0
return_q4 = set()
for ic, i in mostPopular:
    count_q4 += ic
    return_q4.add(i)
    if count_q4 > totalRead*0.75:           # threshold from Q2
        break

ypred_q4 = []
for d in ratingsValid:
    # get similaritiy for all pairs in validation set
    # d[1] = b' = books from training set
    max_sim = 0
    for b in booksPerUser_train[d[0]]:
        if b == d[1]:
            # b = books from validation set, using the user in validation set
            continue
        # compute jaccard similarity between users in train for books in validation set and training set
        sim = Jaccard(usersPerBook_train[b], usersPerBook_train[d[1]])
        if max_sim < sim:
            max_sim = sim
    if max_sim > 0 and d[1] in return_q4:
        ypred_q4.append(1)
    else:
        ypred_q4.append(0)


# %%
acc4 = accuracy(ypred_q4, y_q1)

# %%
answers['Q3'] = acc3
answers['Q4'] = acc4
print(f"acc3: {acc3}")
print(f"acc4: {acc4}")

# %%
assertFloat(answers['Q3'])
assertFloat(answers['Q4'])

# %%
predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    max_sim = 0
    for b2 in booksPerUser_train[u]:
        if b2 == b:
            # b = books from validation set, using the user in validation set
            continue
        # compute jaccard similarity between users in train for books in validation set and training set
        sim = Jaccard(usersPerBook_train[b2], usersPerBook_train[b])
        if max_sim < sim:
            max_sim = sim
    if max_sim > 0 and b in return_q4:
       predictions.write(f"{u},{b},1\n")
    else:
        predictions.write(f"{u},{b},0\n")

predictions.close()

# %%
answers['Q5'] = "I confirm that I have uploaded an assignment submission to gradescope"

# %%
assert type(answers['Q5']) == str

# %%
##################################################
# Category prediction (CSE158 only)              #
##################################################

# %%
### Question 6

# %%
data = []

for d in readGz("train_Category.json.gz"):
    data.append(d)

# %%
data[0]

# %%
category_all_data = [d for d in data]
category_train_data = category_all_data[:90000]
category_valid_data = category_all_data[90000:]

wordCount_train_q6 = defaultdict(int)
totalWords = 0

# %%
punct  = string.punctuation
# stemmer = PorterStemmer()

# %%
# get unique words and its counts
for d in category_train_data:
    rev:string = d['review_text']
    rev = rev.lower()
    rev = [c for c in rev if not (c in punct)]
    rev = ''.join(rev)
    words = rev.strip().split()
    for w in words:
        wordCount_train_q6[w] += 1
        totalWords += 1

# get the top 1000 most common words
counts_q6 = [(wordCount_train_q6[w], w) for w in wordCount_train_q6]
counts_q6.sort()
counts_q6.reverse()

# %%
answers['Q6'] = counts_q6[:10]

# %%
assert [type(x[0]) for x in answers['Q6']] == [int]*10
assert [type(x[1]) for x in answers['Q6']] == [str]*10

# %%
### Question 7

# %%
# get top 1000 unique words 
common1000 = [w for count, w in counts_q6[:1000]]
wordId_q7 = dict(zip(common1000, range(len(common1000))))
word_set_q7 = set(common1000) # build dictionary
len(word_set_q7)


# %%
# iterate through entire dataset and count number of top 1000 unique words per review
def feature(datum):
    # Iterate through a review at a time
    feat = [0]*len(word_set_q7)
    rev = datum['review_text']
    rev = [c for c in rev if not (c in punct)]
    rev = ''.join(rev)
    words = rev.strip().split()
    for w in words:
        # check each word in review
        if not (w in word_set_q7): continue
        feat[wordId_q7[w]] += 1
    feat.append(1)                          # constant 
    return feat

X_q7 = [feature(d) for d in data] # a list_a of list_b. len(X) = len(y)
y = [d['genreID'] for d in data]

print(X_q7[0][:10])
print(y[:10])


# %%
X_q7train = X_q7[:9*len(X_q7)//10]
ytrain = y[:9*len(y)//10]
X_q7valid = X_q7[9*len(X_q7)//10:]
yvalid = y[9*len(y)//10:]

# %%
model = linear_model.LogisticRegression(C=1.0)
model.fit(X_q7train, ytrain)
ypred_valid_q7 = model.predict(X_q7valid)

acc7 = accuracy(yvalid, ypred_valid_q7)

# %%
answers['Q7'] = acc7
print(acc7)

# %%
assertFloat(answers['Q7'])

# %%
### Question 8

# %%
acc8 = acc7
for c in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]:
    model_q8 = linear_model.LogisticRegression(C=c)
    model_q8.fit(X_q7train, ytrain)
    ypred_valid_q8 = model_q8.predict(X_q7valid)
    curr_acc = accuracy(yvalid, ypred_valid_q8)
    if curr_acc > acc8:
        acc8 = curr_acc
        print(f"acc8: {acc8}")
        print(f"c: {c}")

# %%
# def feature(datum, common_words, word_set):
#     # Iterate through a review at a time
#     feat = [0]*len(word_set_q7)
#     rev = datum['review_text']
#     rev = [c for c in rev if not (c in punct)]
#     rev = ''.join(rev)
#     words = rev.strip().split()
#     for w in common1000:
#         # check each word in review
#         if not (w in word_set_q7):
#             continue
#         feat[wordId_q7[w]] += 1
#     feat.append(1)                          # constant
#     return feat


# %%
# for size in [2000, 3000, 4000, 5000, 6000]:
#     # get top {size} unique words
#     common_words = [w for count, w in counts_q6[:size]]
#     wordId_q8 = dict(zip(common_words, range(len(common_words))))
#     word_set_q8 = set(common_words)  # build dictionary
#     len(word_set_q7)
#     for 


# %%
answers['Q8'] = acc8
print(acc8)

# %%
assertFloat(answers['Q8'])

# %%
# Run on test set

# %%
predictions = open("predictions_Category.csv", 'w')
pos = 0

for l in open("pairs_Category.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    # (etc.)

predictions.close()

# %%
f = open("answers_hw3.txt", 'w')
f.write(str(answers) + '\n')
f.close()

# %%




# %%
import numpy
import urllib
import scipy.optimize
import random
from sklearn import linear_model
import gzip
from collections import defaultdict

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
f = open("5year.arff", 'r')

# %%
# Read and parse the data
while not '@data' in f.readline():
    pass

dataset = []
for l in f:
    if '?' in l: # Missing entry
        continue
    l = l.split(',')
    values = [1] + [float(x) for x in l]
    values[-1] = values[-1] > 0 # Convert to bool
    dataset.append(values)

# %%
X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]

print(len(X))
print(len(y))

# %%
answers = {} # Your answers

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
def BER(pred, y):
    """Evaluate Balanced Error rate of prediction and true values"""
    TP_ = numpy.logical_and(pred, y)
    FP_ = numpy.logical_and(pred, numpy.logical_not(y))
    TN_ = numpy.logical_and(numpy.logical_not(pred), numpy.logical_not(y))
    FN_ = numpy.logical_and(numpy.logical_not(pred), y)

    TP = sum(TP_)
    FP = sum(FP_)
    TN = sum(TN_)
    FN = sum(FN_)

    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)

    BER = 1 - (0.5*(TPR + TNR))
    return BER


# %%
### Question 1

# %%
mod = linear_model.LogisticRegression(C=1)
mod.fit(X,y)

pred = mod.predict(X)

# %%
acc1 = accuracy(pred, y)
ber1 = BER(pred, y)


# %%
answers['Q1'] = [acc1, ber1] # Accuracy and balanced error rate

# %%
assertFloatList(answers['Q1'], 2)

# %%
### Question 2

# %%
mod = linear_model.LogisticRegression(C=1, class_weight='balanced')
mod.fit(X,y)

pred = mod.predict(X)

# %%
acc2 = accuracy(pred, y)
ber2 = BER(pred, y)

# %%
answers['Q2'] = [acc2, ber2]

# %%
assertFloatList(answers['Q2'], 2)

# %%
### Question 3

# %%
random.seed(3)
random.shuffle(dataset)

# %%
X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]

# %%
Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]

# %%
len(Xtrain), len(Xvalid), len(Xtest)

# %%
model = linear_model.LogisticRegression(C=1, class_weight='balanced')
model.fit(Xtrain, ytrain)

# %%
pred_train = model.predict(Xtrain)
pred_valid = model.predict(Xvalid)
pred_test = model.predict(Xtest)

# %%
berTrain = BER(pred_train, ytrain)
berValid = BER(pred_valid, yvalid)
berTest = BER(pred_test, ytest)

# %%
answers['Q3'] = [berTrain, berValid, berTest]

# %%
assertFloatList(answers['Q3'], 3)

# %%
### Question 4

# %%


# %%
def make_pred(X, y, c, data_set):
    """Make a prediction using different regularization coefficients"""
    model = linear_model.LogisticRegression(C=c, class_weight='balanced')
    model.fit(X, y)

    pred = model.predict(data_set)
    return pred

# %%
pred_n10000 = make_pred(Xtrain, ytrain, c=10**-4, data_set=Xvalid)
pred_n1000 = make_pred(Xtrain, ytrain, c=10**-3, data_set=Xvalid)
pred_n100 = make_pred(Xtrain, ytrain, c=10**-2, data_set=Xvalid)
pred_n10 = make_pred(Xtrain, ytrain, c=10**-1, data_set=Xvalid)
pred_1 = make_pred(Xtrain, ytrain, c=1, data_set=Xvalid)
pred_10 = make_pred(Xtrain, ytrain, c=10, data_set=Xvalid)
pred_100 = make_pred(Xtrain, ytrain, c=10**2, data_set=Xvalid)
pred_1000 = make_pred(Xtrain, ytrain, c=10**3, data_set=Xvalid)
pred_10000 = make_pred(Xtrain, ytrain, c=10**4, data_set=Xvalid)


# %%
BER_n10000 = BER(pred_n10000, yvalid)
BER_n1000 = BER(pred_n1000, yvalid)
BER_n100 = BER(pred_n100, yvalid)
BER_n10 = BER(pred_n10, yvalid)
BER_1 = BER(pred_1, yvalid)
BER_10 = BER(pred_10, yvalid)
BER_100 = BER(pred_100, yvalid)
BER_1000 = BER(pred_1000, yvalid)
BER_10000 = BER(pred_10000, yvalid)

berList = [BER_n10000, BER_n1000, BER_n100,
           BER_n10, BER_1, BER_10, BER_100, BER_1000, BER_10000]


# %%
answers['Q4'] = berList

# %%
assertFloatList(answers['Q4'], 9)

# %%
### Question 5

# %%
# check for lowest BER from the regularization pipeline 
print(berList)
print(min(berList))

# %%
c_s = [10**-4, 10**-3, 10**-2, 10**-1, 1, 10, 10**2, 10**3, 10**4] # list of regularization coefficients
best_c_ind= berList.index(min(berList))
bestC = c_s[best_c_ind] 

print(f"Best C at index: {best_c_ind}")
print(f"Best C: {bestC}")

# %%
best_pred = make_pred(Xtrain, ytrain, bestC, Xtest) # create model with the best C value(smallest BER) and test data
ber5 = BER(best_pred, ytest)

# %%
answers['Q5'] = [bestC, ber5]

# %%
assertFloatList(answers['Q5'], 2)

# %%
### Question 6

# %%
f = gzip.open("young_adult_10000.json.gz")
dataset = []
for l in f:
    dataset.append(eval(l))

# %%
dataTrain = dataset[:9000]
dataTest = dataset[9000:]

print(dataTrain[0])

# %%
# Some data structures you might want

usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)
ratingDict = {} # To retrieve a rating for a specific user/item pair

for d in dataTrain:
    user = d['user_id']
    item = d['book_id']
    review = d['review_id']
    rating = d['rating']
    usersPerItem[item].add(user) # for each item, there is a set of users that gave a rating
    itemsPerUser[user].add(item) # for each user, there is a set of items they've rated
    reviewsPerUser[user].append(d) # for each user, there is a list of reviews the user made
    reviewsPerItem[item].append(d) # for each item, there is a list of reviews that was made for item
    ratingDict[(user, item)] = rating

# %%
def Jaccard(s1, s2):
    numerator = len(s1.intersection(s2))
    denominator = len(s1.union(s2))
    return numerator/denominator

# %%
def mostSimilar(i, N=10):
    similarities = []        # list of (Jaccard_similarity, item) pair
    users =  usersPerItem[i] # list of users that rated item i 
    for i2 in usersPerItem:
        if i2 == i: continue
        sim = Jaccard(users, usersPerItem[i2]) # compare item that we're looking at to other items, to see if same user have rated it 
        similarities.append((sim, i2))
    similarities.sort(reverse=True)
    
    return similarities[:N]

# %%
answers['Q6'] = mostSimilar('2767052', 10)

# %%
assert len(answers['Q6']) == 10
assertFloatList([x[0] for x in answers['Q6']], 10)

# %%
### Question 7

# %%
# def avg_user_rating(user):
#     ratings = []
#     for rev in reviewsPerUser[user]:
#         ratings.append(rev['rating'])
#     if len(ratings) > 0:
#         avg_rating = sum(ratings)/len(ratings)
#         return avg_rating
#     return 0

def avg_item_rating(item):
    """Get the average rating of an item"""
    ratings = []
    for rev in reviewsPerItem[item]:
        ratings.append(rev['rating'])

    if len(ratings) > 0:
        avg_rating = sum(ratings)/len(ratings)
        return avg_rating
    return 0

# %%
def predRating(user, item):
    ratings = []
    similarities = []
    for d in  reviewsPerUser[user]:
        i2 = d['book_id']
        if item == i2: continue
        ratings.append(d['rating'] - (avg_item_rating(i2)))                # subtract by the average rating for regularization purposes
        similarities.append(Jaccard(usersPerItem[item], usersPerItem[i2])) # Use Jaccard similaritiy to find similarity between items
    if sum(similarities) > 0:
        weightedRatings = [(x*y) for x,y in zip(ratings, similarities)]
        return avg_item_rating(item) + sum(weightedRatings) / sum(similarities)     # uses the form of function presented in Q7
    else:
        return sum([d['rating'] for d in dataset]) / len(dataset)

        

# %%
sse = 0
for d in dataTest:
    sse += (d['rating']-predRating(d['user_id'], d['book_id']))**2

# %%
mse7 = sse/len(dataTest)

# %%
answers['Q7'] = mse7

# %%
assertFloat(answers['Q7'])

# %%
### Question 8

# %%
def predRatingByUser(user, item):
    ratings = []
    similarities = []
    for d in  reviewsPerUser[user]:
        i2 = d['book_id']
        u2 = d['user_id']
        if user == u2: continue
        ratings.append(d['rating'] - (avg_item_rating(i2)))
        similarities.append(Jaccard(itemsPerUser[user], itemsPerUser[u2]))      # USe Jaccard to get similarities between set of items that user rated compared to u2
    if sum(similarities) > 0:
        weightedRatings = [(x*y) for x,y in zip(ratings, similarities)]
        return avg_item_rating(item) + sum(weightedRatings) / sum(similarities)
    else:
        return sum([d['rating'] for d in dataset]) / len(dataset)

# %%
sse = 0
for d in dataTest:
    sse += (d['rating']-predRatingByUser(d['user_id'], d['book_id']))**2

# %%
mse8 = sse/len(dataTest)

# %%
answers['Q8'] = mse8

# %%
assertFloat(answers['Q8'])

# %%
f = open("answers_hw2.txt", 'w')
f.write(str(answers) + '\n')
f.close()

# %%




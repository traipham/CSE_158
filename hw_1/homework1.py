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

# % Helper Functions %
def assertFloat(x): # Checks that an answer is a float
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

def solve_theta(x, y):
    xT = numpy.transpose(x)
    theta = numpy.matmul(numpy.matmul(numpy.linalg.inv(numpy.matmul(xT, x)), xT), y)
    return theta

# % Open file and initiate dataset %
f = gzip.open("young_adult_10000.json.gz")
dataset = []
for l in f:
    dataset.append(json.loads(l))

# % Get dataset len %
len(dataset)

# % Answer dictionary %
answers = {} # Put your answers to each question in this dictionary

# % Look through data %
dataset[0]

### Question 1

# %%
def feature(datum):
    # get number of "!" in review
    feat = datum['review_text'].count("!")
    return [1] + [feat]


# %%
# get count of "!" for each review
X = [feature(d) for d in dataset]
# get rating for book
Y = [d['rating'] for d in dataset]

# %%
print(X[:5])
print(Y[:5])

# %%
theta, residuals, rank, s = numpy.linalg.lstsq(X, Y) # solve for theta using feature matrix and rating vector
print(theta)

theta0 = theta[0]
theta1 = theta[1]

# %%
model = linear_model.LinearRegression(fit_intercept=False)
model.fit(X, Y)

y_pred = model.predict(X)
sse = sum([x**2 for x in (Y-y_pred)])

mse = sse/len(Y)
print(mse)

# %%
answers['Q1'] = [theta0, theta1, mse]

# %%
assertFloatList(answers['Q1'], 3) # Check the format of your answer (three floats)

# %%
### Question 2

# %%
def feature(datum):
    ft_1 = datum["review_text"].count("!")
    ft_2 = len(datum["review_text"])
    return [1] + [ft_2] + [ft_1]

# %%
X = [feature(d) for d in dataset]
Y = [d["rating"] for d in dataset]

print(X[:5])
print(Y[:5])

# %%
theta, residuals, rank, s = numpy.linalg.lstsq(X, Y)
print(theta)

theta0, theta1, theta2 = theta[:3]
print(theta0)
print(theta1)
print(theta2)

# %%

model = linear_model.LinearRegression(fit_intercept=False)
model.fit(X, Y)

y_pred = model.predict(X)
sse = sum([x**2 for x in (Y-y_pred)])

mse = sse/len(Y)
print(mse)

# %%
answers['Q2'] = [theta0, theta1, theta2, mse]

# %%
assertFloatList(answers['Q2'], 4)

# %%
### Question 3

# %%
def feature(datum, deg: int = 5):
    # feature for a specific polynomial degree
    all_feat = [1]
    for exp in range(1,deg+1):
        ft = float((datum["review_text"].count("!"))**exp)
        all_feat += [ft]
    return all_feat



def standarized_w_max_col(ft_list):
    max_vals = []

    tp_ft_list = numpy.transpose(ft_list)
    for ind in range(len(tp_ft_list)):
        max_val = max(tp_ft_list[ind])
        tp_ft_list[ind] = tp_ft_list[ind]/max_val

    return numpy.transpose(tp_ft_list)


# %%
X_1 = [feature(d, 1) for d in dataset]

X_2 = [feature(d, 2) for d in dataset]

X_3 = [feature(d, 3) for d in dataset]

X_4 = [feature(d, 4) for d in dataset]

X_5 = [feature(d) for d in dataset]

Y = [d["rating"] for d in dataset]

print(X_1[:10])
print(X_2[:5])
print(X_5[:5])
# print(numpy.transpose(X_5)[2][:5])
print(Y[:5])

# %%
theta_5, residuals, rank, s = numpy.linalg.lstsq(X_5, Y)
print(theta_5)


# %%
model_1 = linear_model.LinearRegression(fit_intercept=False)
model_2 = linear_model.LinearRegression(fit_intercept=False)
model_3 = linear_model.LinearRegression(fit_intercept=False)
model_4 = linear_model.LinearRegression(fit_intercept=False)
model_5 = linear_model.LinearRegression(fit_intercept=False)

model_1.fit(X_1, Y)
model_2.fit(X_2, Y)
model_3.fit(X_3, Y)
model_4.fit(X_4, Y)
model_5.fit(X_5, Y)

y_pred_1 = model_1.predict(X_1)
y_pred_2 = model_2.predict(X_2)
y_pred_3 = model_3.predict(X_3)
y_pred_4 = model_4.predict(X_4)
y_pred_5 = model_5.predict(X_5)

sse_1 = sum([x**2 for x in (Y-y_pred_1)])
sse_2 = sum([x**2 for x in (Y-y_pred_2)])
sse_3 = sum([x**2 for x in (Y-y_pred_3)])
sse_4 = sum([x**2 for x in (Y-y_pred_4)])
sse_5 = sum([x**2 for x in (Y-y_pred_5)])

mse_1 = sse_1/len(Y)
mse_2 = sse_2/len(Y)
mse_3 = sse_3/len(Y)
mse_4 = sse_4/len(Y)
mse_5 = sse_5/len(Y)

mses = [mse_1] + [mse_2] + [mse_3] + [mse_4] + [mse_5]
print(mses)


# %%
answers['Q3'] = mses

# %%
assertFloatList(answers['Q3'], 5)# List of length 5

# %%
### Question 4

# %%
train_set_1 = X_1[:len(X_1)//2]
train_set_2 = X_2[:len(X_2)//2]
train_set_3 = X_3[:len(X_3)//2]
train_set_4 = X_4[:len(X_4)//2]
train_set_5 = X_5[:len(X_5)//2]
y_train = Y[:len(Y)//2]

test_set_1 = X_1[len(X_1)//2:]
test_set_2 = X_2[len(X_2)//2:]
test_set_3 = X_3[len(X_3)//2:]
test_set_4 = X_4[len(X_4)//2:]
test_set_5 = X_5[len(X_5)//2:]
y_test = Y[len(Y)//2:]

# %%
def create_model(x, y):
    model = linear_model.LinearRegression(fit_intercept=False)
    model.fit(x,y)
    return model

def calculate_mse(model, set, y):
    y_pred = model.predict(set)
    sse = sum([x**2 for x in (y-y_pred)])

    mse = sse/len(y)
    return mse

# %%
# trained model
q4_model_1 = create_model(train_set_1, y_train)
q4_model_2 = create_model(train_set_2, y_train)
q4_model_3 = create_model(train_set_3, y_train)
q4_model_4 = create_model(train_set_4, y_train)
q4_model_5 = create_model(train_set_5, y_train)
# mse of test set
q4_mse_1_tst = calculate_mse(q4_model_1, test_set_1, y_test)
q4_mse_2_tst = calculate_mse(q4_model_2, test_set_2, y_test)
q4_mse_3_tst = calculate_mse(q4_model_3, test_set_3, y_test)
q4_mse_4_tst = calculate_mse(q4_model_4, test_set_4, y_test)
q4_mse_5_tst = calculate_mse(q4_model_5, test_set_5, y_test)

mses = [q4_mse_1_tst] + [q4_mse_2_tst] + [q4_mse_3_tst] + [q4_mse_4_tst] + [q4_mse_5_tst]
print(mses)


# %%
answers['Q4'] = mses

# %%
assertFloatList(answers['Q4'], 5)

# %%
### Question 5

# %%
Y = [d["rating"] for d in dataset]
mean =  sum(Y)/len(Y) # for trivial prediction, the mean of result/output data is the best predictor (prediction set)
print(f"mean={mean}")


mae = sklearn.metrics.mean_absolute_error(
    y_true=y_test, y_pred=[mean for i in range(len(y_test))])
print(mae)

# %%
answers['Q5'] = mae

# %%
assertFloat(answers['Q5'])

# %%
### Question 6

# %%
f = open("beer_50000.json")
dataset = []
for l in f:
    if 'user/gender' in l:
        dataset.append(eval(l))

# %%
len(dataset)

# %%
dataset[0]

# %%
X = [[1] + [d["review/text"].count("!")] for d in dataset]
# X_train = X[:len(X)//2]
# X_test = X[len(X)//2:]

y_f = ['Female' in d['user/gender'] for d in dataset]
# y_f_train = y_f[:len(y_f)//2]
# y_f_test = y_f[len(y_f)//2:]

# %%
model = linear_model.LogisticRegression(C=1.0)
model.fit(X, y_f)

pred = model.predict(X)
print(pred)

# %%
TP_ = numpy.logical_and(pred, y_f)
FP_ = numpy.logical_and(pred, numpy.logical_not(y_f))
TN_ = numpy.logical_and(numpy.logical_not(pred), numpy.logical_not(y_f))
FN_ = numpy.logical_and(numpy.logical_not(pred), y_f)

TP = sum(TP_)
FP = sum(FP_)
TN = sum(TN_)
FN = sum(FN_)

TPR = TP / (TP + FN)
TNR = TN / (TN + FP)

BER = 1 - 0.5*(TPR + TNR)


# %%
answers['Q6'] = [TP, TN, FP, FN, BER]

# %%
assertFloatList(answers['Q6'], 5)

# %%
### Question 7

# %%
model = linear_model.LogisticRegression(C=1.0, class_weight='balanced')
model.fit(X, y_f)
pred = model.predict(X)

TP_ = numpy.logical_and(pred, y_f)
FP_ = numpy.logical_and(pred, numpy.logical_not(y_f))
TN_ = numpy.logical_and(numpy.logical_not(pred), numpy.logical_not(y_f))
FN_ = numpy.logical_and(numpy.logical_not(pred), y_f)

TP = sum(TP_)
FP = sum(FP_)
TN = sum(TN_)
FN = sum(FN_)

TPR = TP / (TP + FN)
TNR = TN / (TN + FP)

BER = 1 - 0.5*(TPR + TNR)

print(TP)
print(FP)
print(TN)
print(FN)
print(BER)


# %%
answers["Q7"] = [TP, TN, FP, FN, BER]

# %%
assertFloatList(answers['Q7'], 5)

# %%
### Question 8

# %%
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(~x))


# %%
X = numpy.reshape([d["review/text"].count("!") for d in dataset], (-1,1))

model = linear_model.LogisticRegression(C=1.0, class_weight='balanced')
model.fit(X, y_f)

pred = model.predict(X)
curr = list(map(sigmoid, pred))
print(curr[:25])
# theta, residuals, rank, s = numpy.linalg.lstsq(X, y_f)
# print(theta)
# print(numpy.matmul(theta,X[23]))
# print(pred[:25])

# %%
def sort_label(prev_v2, y_output):
    conf_arr = numpy.ndarray.tolist(model.decision_function(prev_v2)) # confidence list
    conf_w_y = [[y_output[i]] + [conf_arr[i]] for i in range(len(y_output))] # confidence with y_vector

    sorted_conf_w_y = sorted(conf_w_y, key= lambda x: x[1], reverse=True)
    res_list = [li[0] for li in sorted_conf_w_y]
        
    return res_list

# %%
k = [1, 10, 100, 1000, 10000]
sorted_label = sort_label(numpy.reshape(pred, (-1,1)), y_f) # get sorted labels

num_rel_ret = [sum(sorted_label[:k_ind]) for k_ind in k]

precisionList = [num_rel_ret[ind]/k[ind] for ind in range(len(k))]
print(precisionList)

# %%
answers['Q8'] = precisionList

# %%
assertFloatList(answers['Q8'], 5) #List of five floats

# %%
f = open("answers_hw1.txt", 'w') # Write your answers to a file
f.write(str(answers) + '\n')
f.close()

# %%




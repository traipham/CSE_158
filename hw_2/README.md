# Homework 2:
You can download the datasets here: https://cseweb.ucsd.edu/classes/fa22/cse258-a/files/homework2.pdf
## Instructions:
### Tasks — Model Pipelines and Diagnostics:
In the first homework, we began to explore a coulpe of issues with the classifiers we built. Namely (1) the
data were not shuffled, and (2) the labels were highly imbalanced. Both of these made it difficult to effectively
build an accurate classifier. Here we’ll try and correct for those issues using the Bankruptcy dataset.
1. Download and parse the bankruptcy data. We’ll use the 5year.arff file. Code to read the data is
available in the stub. Train a logistic regressor (e.g. sklearn.linear model.LogisticRegression) with
regularization coefficient C = 1.0. Report the accuracy and Balanced Error Rate (BER) of your classifier.
2. Retrain the above model using the class weight=’balanced’ option. Report the accuracy and BER of
your new classifier.
3. Shuffle the data, and split it into training, validation, and test splits, with a 50/25/25% ratio. Use the
code in the stub provided to ensure that your random split is the same as the reference
solution. Using the class weight=’balanced’ option, and training on the training set, report the
training/validation/test BER.
4. Implement a complete regularization pipeline with the above classifier. Consider values of C in the range
{10−4
, 10−3
, . . . , 103
, 104}. Report the validation BER for each value of C.
5. Based on these values, which classifier would you select (in terms of generalization performance)? Report
the best value of C and its performance (BER) on the test set.
1
### Tasks — Recommendation:
For this question we’ll use the Goodreads book review data. The first 90% of the data should be used for
training and the remaining 10% for evaluation (the stub shows how to split the data).

6. Which 10 items have the highest Jaccard similarity compared to the first item (i.e., the item from the first
review, ‘2767052’)? Report both similarities and item IDs (your answer should be a list of (similarity,
item id) tuples). Note that the test data should not be used for this question.
7. Implement a rating prediction model based on the similarity function
,
(there is already a prediction function similar to this in the provided example code, you can either start
from scratch or modify an existing solution). Report the MSE (on the test set) of this rating prediction
function when Sim(i, j) = Jaccard(i, j).1
1. Modify the similarity function from Question 7 to interchange users and items (i.e., in terms of the
similarity between users Sim(u, v) rather than Sim(i, j)), and report its MSE on the test data.
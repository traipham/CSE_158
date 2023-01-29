# Homework 1:
You can download the datasets here: https://cseweb.ucsd.edu/classes/fa22/cse258-a/files/homework1.pdf
## Homework 1 Task: Regression
1. Train a simple predictor that estimates a rating from the number of times the exclamation mark (!)
symbol is used in the review. Report the weights and MSE of model
2. Re-train your predictor so as to include a second feature based on the length (in characters). Report the weights and MSE of model
3. Train a model that fits a polynomial function to estimate ratings based on our ‘!’ feature. Report MSE
4. Repeat the above question, but this time split the data into a training and test set. You should split the
data into 50%/50% train/test fractions. The first half of the data should be in the training set and the
second half in the test set.1 Report the MSE of each model on the test set.
5. Given a trivial predictor, i.e., y = θ0, what is the best possible predictor (i.e., value of θ0) in terms of the
Mean Absolute Error (MAE)? For your answer report the MAE of your predictor on the test set from the previous question.

## Homework 1 Task: Classification
In this question, using the beer review data, we’ll try to predict user gender based on users’ beer reviews. Load
the 50,000 beer review dataset, discarding any entries that don’t include a specified gender.
1. Fit a logistic regressor that estimates gender from the number of ‘!’ characters, i.e.,
p(gender is female) ≃ σ(θ0 + θ1 × [number of !])
Report the number of True Positives, True Negatives, False Positives, False Negatives, and the Balanced
Error Rate of the predictor (your answer should be a list of length 5). You may use a logistic regression
library with default parameters, e.g. linear model.LogisticRegression() from sklearn.
2. Retrain the regressor using the class weight=’balanced’ option, and report the same error metrics as
above.
3. Report the precision@K of your balanced classifier for K ∈ [1, 10, 100, 1000, 10000] (your answer should
be a list of five precision values).
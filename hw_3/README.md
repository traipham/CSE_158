# Homework 3:
You can download the datasets here: https://cseweb.ucsd.edu/classes/fa22/cse258-a/files/homework3.pdf
## Instructions:
### Tasks (Read prediction)
Since we don’t have access to the test labels, we’ll need to simulate validation/test sets of our own.
So, let’s split the training data (‘train Interactions.csv.gz’) as follows:
(1) Reviews 1-190,000 for training
(2) Reviews 190,001-200,000 for validation
(3) Upload to Gradescope for testing only when you have a good model on the validation set. If you can build
such a validation set correctly, it will significantly speed up your testing and development time.
1. Although we have built a validation set, it only consists of positive samples. For this task we also need
examples of user/item pairs that weren’t read. For each (user,book) entry in the validation set, sample a
negative entry by randomly choosing a book that user hasn’t read.1 Evaluate the performance (accuracy)
of the baseline model on the validation set you have built.
2. The existing ‘read prediction’ baseline just returns True if the item in question is ‘popular,’ using a
threshold based on those books which account for 50% of all interactions (totalRead/2). Assuming that
the ‘non-read’ test examples are a random sample of user-book pairs, this threshold may not be the best
one. See if you can find a better threshold (or otherwise modify the thresholding strategy); report the
new threshold and its performance of your improved model on your validation set.
3. A stronger baseline than the one provided might make use of the Jaccard similarity (or another similarity
metric). Given a pair (u, b) in the validation set, consider all training items b
′
that user u has read. For
each, compute the Jaccard similarity between b and b
′
, i.e., users (in the training set) who have read
b and users who have read b
′
. Predict as ‘read’ if the maximum of these Jaccard similarities exceeds a
threshold (you may choose the threshold that works best). Report the performance on your validation
set.
1This is how I constructed the test set; a good solution should mimic this procedure as closely as possible so that your leaderboard
performance is close to their validation performance.
1
4. Improve the above predictor by incorporating both a Jaccard-based threshold and a popularity based
threshold. Report the performance on your validation set.2
5. To run our model on the test set, we’ll have to use the files ‘pairs Read.csv’ to find the userID/bookID
pairs about which we have to make predictions. Using that data, run the above model and upload your
solution to Gradescope. If you’ve already uploaded a better solution, that’s fine too! Your answer should
be the string “I confirm that I have uploaded an assignment submission to gradescope”.
### Tasks (Category prediction)
Please try to run these experiments using the specified training data and dictionary size; an efficient solution
should take only a few seconds to run. Please only use a smaller dictionary size (i.e., fewer words), or a smaller
training set size, if the experiments are taking too long to run.

6. Using the review data (train Category.json.gz), build training/validation sets consisting of 90,000/10,000
reviews. We’ll start by building features to represent common words. Start by removing punctuation
and capitalization, and finding the 1,000 most common words across all reviews (‘review text’ field) in
the training set. See the ‘text mining’ lectures for code for this process. Report the 10 most common
words, along with their frequencies (your answer should be a list of 10 (frequency, word) tuples).
7. Build bag-of-words feature vectors by counting the instances of these 1,000 words in each review. Set the
labels (y) to be the ‘genreID’ column for the training instances. You may use these labels directly with
sklearn’s LogisticRegression model, which will automatically perform multiclass classification. Report
performance (accuracy) on your validation set.
8. Try to improve upon the performance of the above classifier by using different dictionary sizes, or changing
the regularization constant C passed to the logistic regression model. Report the performance of your
solution, and upload your prediction file to the Assignment 1 gradescope.

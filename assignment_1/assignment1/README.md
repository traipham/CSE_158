# Assignment 1
Download datasets here: https://cseweb.ucsd.edu/classes/fa22/cse258-a/files/assignment1.pdf
## Instructions:
In this assignment you will build recommender systems to make predictions related to book reviews from
Goodreads. Tasks to accomplish:
1. Perform predictions given a (user,book) pair from ‘pairs Read.csv’ whether the user
would read the book (0 or 1). Accuracy will be measured in terms of the categorization accuracy (fraction
of correct predictions). The test set has been constructed such that exactly 50% of the pairs correspond
to read books and the other 50% do not.
2. Predict the category of a book from a review. Five categories are used
for this task, which can be seen in the baseline program, namely Children’s, Comics/Graphic Novels,
Fantasy, Mystery/Thriller, and Romance. Performance will be measured in terms of the fraction of
correct classifications.

## Assignment 1 Brainstorm: Would read prediction task
- Find similarities between books that user would read and seeing if it's in most popular book list(current)
- Find if it's in most popular book list (based on ranking) (not good)
- Find if user is in most popular user list (baed on number of books user have read) (not good)
- Used Bayesian ranking tensor flow implementation
  - used implicit implemntation (cam across error)
  - try training it on training data
  - it looks like rating/read bool is not kept in mind in the bayesian prediction
  - Not sure about any other approach


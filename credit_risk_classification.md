# Classification Modeling

## Overview

This project trains and evaluates a logistic regression model based on loan risk. It uses a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.

More specifically, the prediction dataset included information on loan sizes and interest rates, loan status, borrowers' income and debt, borrowers' debt-to-income ratios,  borrowers' number of financial accounts and any associated derogatory marks.

The targeted values for the model were imbalanced, with 75,036 individuals labeled as having a "healthy loan" status and 2,500 being "high-risk."

The machine learning process invovled:
* Using train_test_split to create unique data sets for training the model separate from the testable data
* Assigning a random_state of 1 so others may duplicate this process and yield similar results
* Fitting a logistic regression model using the training data
* Testing the logistic regression model using the testing data
* Evaluating the model's performance by calculating the accuracy score, generating a confusion matrix, and printing a classification report

## Results

Here are descriptions of key machine learning model evaluation concepts and the above model's results:

* *Accuracy:* Evaluates the number of correct predictions as a percentage of the number of observations n the dataset (out of 100%).  In this case, the model classifies 95% into the correct class.
* *Precision:* The ratio of correctly predicted positive observations to the total predicted positive observations. Out of 100 applicants predicted to be "high-risk," only 85 are actually "high risk."
* *Recall:* The ratio of correctly predicted positive observation to all predicted observations for that class.  Out of 100 "high-risk" applicants, the model predicts 91 of them correctly.
* *F1:* The weighted harmonic mean of precision and recall, such that the best score is 1.0 and the worst is 0.0. Out of 100 "healthy loan" applicants, the model predicts 100 of them correctly.

## Summary

* This model performs fairly well in accurately spotting "healthy loan" and "high-risk" applicants, though there is some room for improvement.
* Specifically, the model could use most improvement in predicting "high-risk" applicants, given the potential for financial loss due to an incorrect prediction.
* To increase application processing efficiency, this model is recommended for assisting loan application reviews, with trained human reviewers double-checking the model's output.

## Sources
UC Berkeley Data Analytics Bootcamp
https://stephenallwright.com/accuracy-vs-balanced-accuracy/
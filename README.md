# Categorizing News Articles on Political Bias

UE24CS352A Machine Learning, Mini-Project (Section D, Team 16, Problem 57)

| Member | SRN |
|---|---|
| Sonu Akash Soundararajan | PES1UG24AM908 |
| Rahul Saran Sharma | PES1UG24AM218 |

## Problem

Given a news article, predict its political leaning. We follow the Stanford CS229 (Spring 2021) project
"Categorizing News Articles on Political Bias" by Crain and Erhan, which trains classifiers on tweets
from members of the US House of Representatives (labelled by the author's party) and then uses them to
label news articles as Democratic or Republican leaning.

Reference paper: https://cs229.stanford.edu/proj2021spr/report2/81996990.pdf

## Plan

1. Reproduce the paper: Bernoulli and Multinomial Naive Bayes (from scratch) and a linear SVM, trained on
   117th Congress tweets (January to April 2021), evaluated on held-out tweets and on news articles.
2. Evaluate fairly: split tweets by congressperson rather than by random rows, and tune hyperparameters
   with cross-validation on the training data only.
3. Extend the baseline with our own improvements and measure each one.

## Data sources

- Tweets: [alexlitel/congresstweets](https://github.com/alexlitel/congresstweets) (daily JSON archive)
- Party labels: [alexlitel/congresstweets-automator](https://github.com/alexlitel/congresstweets-automator) (`historical-users-filtered.json`)
- News articles: [ramybaly/Article-Bias-Prediction](https://github.com/ramybaly/Article-Bias-Prediction)
  (Baly et al., EMNLP 2020; AllSides labels). Used in place of the paper's 100 poll-labelled articles,
  which were never published.

## Status

Work in progress. Setup and run instructions will be added here as the code lands.

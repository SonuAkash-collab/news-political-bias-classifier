# Categorizing News Articles on Political Bias

**Team 16, Section D:** Sonu Akash Soundararajan (PES1UG24AM908), Rahul Saran Sharma (PES1UG24AM218)

**Course:** UE24CS352A Machine Learning, Mini-Project, Problem 57

**Reference paper:** D. Crain and Y. Erhan, "Categorizing News Articles on Political Bias", Stanford CS229, Spring 2021

## Problem statement

The task is to predict whether a news article leans Democratic or Republican. Labelled articles are
expensive to collect, so Crain and Erhan (2021) train on tweets by members of the US House of
Representatives, each labelled with its author's party, and compare Bernoulli and Multinomial Naive
Bayes with a linear SVM on held-out tweets and on news articles.

We reproduce the paper under a stricter evaluation. The paper splits tweets at random, so every
congressperson appears in both train and test and a model can score by recognising people (handles,
district names, pet phrases) rather than party language. Its SVM constant C = 0.06 was chosen on the
tweet test set. Its news articles were labelled by a small poll and never released. We measure how much
the random split inflates accuracy, tune every hyperparameter without touching a test set, replace the
private articles with a large public dataset, and test three changes aimed at the gap between tweets
and articles.

## Dataset

**Tweets.** We use Alex Litel's congresstweets archive with the paper's date range, 3 January to
1 April 2021 (the start of the 117th Congress), and take party labels from the member list of the
companion congresstweets-automator project. We keep House members of party D or R, attach each tweet to
its author through the account handle, and drop retweets and repeated texts. One member can own an
office and a campaign account, so a person is identified by the Bioguide ID, not the handle. This leaves
121,976 tweets from 450 members. Most are Democratic: always predicting D is right on 64.0% of a random
20% test sample.

**News articles.** In place of the paper's private articles we use the Article-Bias-Prediction dataset
of Baly et al. (EMNLP 2020), labelled left, center or right by AllSides. Its media-based split keeps each
news outlet in only one of train, validation and test, so a model cannot score by recognising the
outlet. To match the paper's two classes we map left to D and right to R and drop center. The binary
test set has 1,001 articles, 402 left and 599 right, so it leans the opposite way from the tweets:
always predicting R is right on 59.8% of articles, always predicting D on only 40.2%.

## Approach

**Features and models.** As in the paper we build three vocabularies: all words as they are,
punctuation removed, and punctuation, hashtags, @-mentions and links removed. Bernoulli Naive Bayes
uses word existence; Multinomial Naive Bayes and the SVM use word counts. Both Naive Bayes models are
our own NumPy code with Laplace smoothing, computed in log space and checked against scikit-learn. The
SVM is scikit-learn's LinearSVC, kept at the paper's C = 0.06 when reproducing the paper.

**Fair evaluation.** We split the tweets 80/20 twice with seed 42. The random split, as in the paper,
leaves 441 members with tweets on both sides. The grouped split holds out whole people: 360 members
(99,273 tweets) for training and 90 others (22,703 tweets) for testing. The gap between the two scores
is the accuracy gained by recognising authors. For example, the token "az01" (an Arizona district)
appears in 322 Democratic tweets and no Republican ones, but only 2 members use it. Hyperparameters are
tuned with 5-fold GroupKFold over the training members only. We report accuracy and macro F1 next to the
majority baseline, since with 61.2% D tweets in the grouped test set a model that never predicts R
reaches 61.2% accuracy but only 0.380 macro F1.

**Experiments.** Experiment 1 repeats the tweet test for the three models and three vocabularies on
both splits. Experiment 2 ranks words by the Bernoulli Naive Bayes log ratio
log P(word | D) / P(word | R) and records how many members used each word, to separate party language
from personal markers. Experiment 3 trains on all tweets and predicts the 1,001 test articles.

**Extensions.** Extension 1 uses TF-IDF weights on word unigrams and bigrams with logistic regression.
TF-IDF lowers the weight of words every politician uses, and bigrams capture phrases such as "voting
rights" whose single words both parties use. C, the minimum document frequency and class weighting are
chosen by grouped cross-validation on macro F1. Extension 2 splits each article into tweet-sized chunks,
scores each chunk with a tweet-trained model and averages the probabilities, to reduce the length and
style mismatch. Extension 3 trains on the articles' own training split, to measure how much of the
article error comes from training on tweets.

## Implementation

## Results

## Conclusions

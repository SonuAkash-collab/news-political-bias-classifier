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
tweet test set. Its news test set is 100 articles labelled by a poll, which were never released. We measure how much
the random split inflates accuracy, tune every hyperparameter without touching a test set, replace the
private articles with a large public dataset, and test three changes aimed at the gap between tweets
and articles.

## Dataset

**Tweets.** We use Alex Litel's congresstweets archive with the paper's date range, 3 January to
1 April 2021 (the start of the 117th Congress), and take party labels from the member list of the
companion congresstweets-automator project. We keep House members of party D or R, attach each tweet to
its author through the account handle, and drop retweets and repeated texts. One member can own an
office and a campaign account, so a person is identified by the Bioguide ID, not the handle. This leaves
121,976 tweets from 450 members: 78,136 Democratic and 43,840 Republican, so 64.1% of tweets are D.

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

The system is implemented as a modular Python package under src/polibias with a reproducible execution pipeline. The package relies on NumPy and scikit-learn for numeric and model routines, with fixed random seed 42 throughout.

Data loading modules process 121,976 Congress tweets and the 30,246 AllSides articles that the media-based split assigns to train, validation or test (of 37,554 in the release) into standardized CSV tables. Naive Bayes classifiers (Bernoulli and Multinomial) are written directly from scratch using NumPy log-space operations with Laplace add-one smoothing, verified against scikit-learn for numeric equivalence. Linear models (LinearSVC, Logistic Regression, Complement Naive Bayes) use scikit-learn.

Hyperparameter tuning uses GroupKFold cross-validation (5 folds, grouped on member Bioguide ID) exclusively within the training partition, selecting on macro F1. For inference, models and feature vectorizers are saved as compressed joblib bundles under models/. An end-to-end runner in scripts/run_all.py coordinates data preparation, model training, metric evaluation, and output generation into results/results.json and results/figures/. Interactive inspection is provided by a Streamlit web interface in app/streamlit_app.py and a terminal fallback in app/cli.py.

## Results

Table 1 summarizes classification performance across held-out tweets and the 1,001 binary news articles under the media-based split (402 Left, 599 Right). The majority baseline always predicts the most frequent class: Democratic on tweets (61.2% accuracy, 0.380 macro F1) and Republican on articles (59.8% accuracy, 0.374 macro F1).

| Task and evaluation setting | Model | Accuracy | Macro F1 | Paper comparison |
|---|---|---|---|---|
| Tweets, random split (80/20) | Bernoulli NB (no_punct) | 85.3% | 0.832 | 86.7% (90/10 split) |
| Tweets, random split (80/20) | Multinomial NB (no_punct) | 85.4% | 0.833 | Not reported |
| Tweets, random split (80/20) | LinearSVC (C=0.06) | 86.3% | 0.847 | Approx 87% |
| Tweets, grouped split (held-out members) | Bernoulli NB (no_punct) | 78.6% | 0.757 | −6.7 pp vs random |
| Tweets, grouped split (held-out members) | Multinomial NB (no_punct) | 78.9% | 0.763 | −6.4 pp vs random |
| Tweets, grouped split (held-out members) | LinearSVC (C=0.06) | 81.2% | 0.795 | −5.1 pp vs random |
| Tweets, grouped split (Extension 1) | TF-IDF + Logistic Regression | 83.3% | 0.824 | +2.1 pp over SVC |
| Articles, cross-domain (Exp 3) | Majority baseline (always R) | 59.8% | 0.374 | Baseline |
| Articles, cross-domain (Exp 3) | Bernoulli NB | 60.6% | 0.590 | 58.0% (100 poll arts) |
| Articles, cross-domain (Exp 3) | Multinomial NB | 60.5% | 0.554 | 55.0% (100 poll arts) |
| Articles, cross-domain (Exp 3) | LinearSVC (tuned C=0.10) | 58.4% | 0.574 | 63.0% (100 poll arts) |
| Articles, chunk voting (Extension 2) | Bernoulli NB (50-word chunks) | 60.8% | 0.571 | +0.2 pp over whole |
| Articles, chunk voting (Extension 2) | LinearSVC (50-word chunks) | 51.8% | 0.513 | −6.6 pp over whole |
| Articles, in-domain (Extension 3) | LinearSVC (tuned C=0.10) | 66.5% | 0.658 | +6.7 pp over baseline |

Holding out whole members in the grouped split reduces tweet test accuracy by 5.0 percentage points on average across the three models and three vocabulary variants (range 2.1 to 6.8). Models on random splits rely heavily on individual author vocabulary rather than party-wide ideology.

On news articles, tweet-trained models barely compete with the 59.8% majority baseline. Bernoulli NB scores 60.6% and Multinomial NB scores 60.5%, while LinearSVC scores 58.4% (below baseline). In Extension 2, sentence-chunk voting with 50-word chunks does not fix this domain gap: chunked Bernoulli NB achieves 60.8%, while chunked LinearSVC drops to 51.8% (58.4% on whole articles).

In contrast, Extension 3 trains directly on AllSides articles under the media-based split. LinearSVC reaches 66.5% accuracy and 0.658 macro F1 (+6.7 percentage points over majority baseline). This demonstrates that the bottleneck in cross-domain political classification is vocabulary and stylistic distribution shift between social media posts and journalistic prose.

## Conclusions

Our reproduction and extension yield three conclusions:

1. Random train-test splits on social media text severely overestimate generalization. Grouping by author drops accuracy by 5.0 percentage points on average (range 2.1 to 6.8) because models cannot exploit member-specific names, handles, or local districts.
2. Cross-domain transfer from Congress tweets to news articles fails. Tweet-trained models achieve 57.7% to 60.6% accuracy on 1,001 AllSides articles, indistinguishable from the 59.8% majority baseline. Sentence-chunk aggregation does not resolve this stylistic mismatch.
3. In-domain article supervision with outlet-disjoint splits achieves 66.5% accuracy (+6.7 percentage points over baseline). Future improvements should focus on domain-adaptive pretraining and entity-masked representations rather than pure surface n-gram transfers.

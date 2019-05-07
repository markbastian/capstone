# Machine Learning Engineer Nanodegree
## Capstone Proposal
Mark Bastian  
April 24<sup>th</sup>, 2019

## Political Tweet Classification

### Domain Background
Twitter is a common platform for people, including politicians, to express their views. Often tweets are highly polarized and opinionated, especially given the current political climate in the United States as well as the [rise of fake "Trolling" tweets](https://www.vox.com/2018/10/19/17990946/twitter-russian-trolls-bots-election-tampering). Given this, it is valuable to have tools to answer such questions as:

* Is this a tweet from a real person?
* What are the political leanings of a tweet?
* Can I determine the party of the author of a tweet?

Besides being interesting, such a tool could be useful for doing things such as targeted advertising of candidates to twitter users in general. Users that express sentiments that correlate well to one party would be good targets for advertising for issues or candidates with similar sentiments and values.

A massive amount of related work exists, including:

* [Actionable and Political Text Classification using Word
Embeddings and LSTM](https://arxiv.org/pdf/1607.02501v2.pdf)
* [Leveraging Deep Learning for Political Leaning Classification](https://medium.com/@sofronije/leveraging-deep-learning-for-political-leaning-classification-4ddf9d1d2f53)
* [On Classifying the Political Sentiment of Tweets](https://pdfs.semanticscholar.org/fe57/b7e7e2d228f02ad60f419e049ce2d223eb86.pdf)
* [Text Classifiers for Political Ideologies](https://nlp.stanford.edu/courses/cs224n/2009/fp/7.pdf)
* [Topic-centric Classification of Twitter
User’s Political Orientation](http://terrierteam.dcs.gla.ac.uk/publications/fang2015sigir.pdf)

### Problem Statement
The problem I wish to solve is classification of the author of a tweet into one of the two major US political parties (Republican or Democrat). Specifically, the model or models will take a tweet or tweet length text as input and produce a prediction of whether the tweet's author is a Democrat or Republican. The input is expected to be political in nature, so I will not detect non-political tweets and I will limit the output categories to only the two major parties.

The dataset I will train on is sourced such that I know beforehand the party of the tweet. This can be broken up into testing and training sets to measure goodness of model fit.

### Datasets and Inputs
I will use the Democrat Vs. Republican Tweets dataset [found here](https://www.kaggle.com/kapastor/democratvsrepublicantweets). 

This dataset provides 86,460 tweets divided roughly evenly (over 42,000 tweets per party) and is sufficiently large to produce sizeable testing and training sets. As the authors are all politicans it is implicitly categorized by the author's political affiliation.

### Solution Statement
I would like to create two models and compare the results. One will use a standard text classification approach from sklearn (e.g. a Bayesian Classifier) and the second will use content from the second half of the course (a deep neural network, CNN, or RNN - not covered in the course).

### Benchmark Model
As part of my solution, my benchmark model will be a Naive Bayes Classifier. My hope is that a neural network will produce better results. [Similar solutions on Kaggle produce accuracies of about 74%.](https://www.kaggle.com/chrislit/dem-or-gop-naive-bayes-classifier)

### Evaluation Metrics
The model will be evaluated using an 80/20 split of the data into training and testing data. Standard metrics such as precision, recall, f1 score, and a confusion matrix can be used to determine how good the resulting model is.

### Project Design
From my research, I believe two possible network encodings that could produce good results would be LSTM/RNNs or CNNs. LSTMs were not covered in the class, but many resources exist to [explain them](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) as well as [describe their use for text classification](https://medium.com/@sabber/classifying-yelp-review-comments-using-lstm-and-word-embeddings-part-1-eb2275e4066b). Additional resources can be found [here](https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17), [here](https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/), or [here](https://www.tensorflow.org/alpha/tutorials/text/text_classification_rnn). Although LSTMs seem like a natural choice for this problem, recent work has show that CNNs [also work well for text problems](https://medium.com/@TalPerry/convolutional-methods-for-text-d5260fd5675f). CNNs also can be trained in parallel, whereas LSTM and RNN training is a linear process so is much slower.

My planned workflow is to:

1. Load the data and create an 80/20 train/test sample set.
2. Create a simple Naive Bayes Classifier as a first model.
3. Create a CNN based classifier as a second model.
4. At this point, I expect quite a bit of tuning will be required on the CNN (e.g. number of layers, size of layers, etc.). I also expect it will take some work and learning to get input text encoded correctly. For example, I know I'll need to master variable-length input encoding to make this work but I haven't fully researched the topic yet.
5. Once I'm happy with the models, I'll report various metrics to compare them (e.g. accuracy, precision, recall, f1).
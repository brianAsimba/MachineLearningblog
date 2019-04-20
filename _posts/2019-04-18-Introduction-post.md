---
layout: post
title: Introduction to Machine Learning
date: 2019-04-18
excerpt: Supervised Machine Learning
tags:
  - Machine Learning
  - Supervised
  - Unsupervised
comments: true
published: true
---

## What is Machine Learning

Machine learning is the science and art of programming computers to learn from data. Machine Learning has been a hot topic lately. This is because of the applications currently utilizing Machine Learning. Every day we use Machine Learning in some shape pf form without knowing. Some of the examples are Amazon recommender systems, google searching algorithm, spam detection in emails, credit card fraudulent detection, mailing addresses algorithm reader, customer segmentation and lastly self-driving to name a few. According to Tom Mitchell, a computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by, improves with experience E.

## Types of Machine Learning
There are various types of Machine Learning. From the diagram below, you can see the main branches of Machine Learning:

    
<img src="https://brianasimba.github.io/MLblog//images/ML_branches.png" style="display: block; margin: auto;" />
  
  
Machine Learning, Robotics and Control theory is a branch of Artificial Intelligence. Machine Learning is further divided mainly into supervised, unsupervised, semi-supervised and reinforcement learning.

## Supervised Learning
Supervised learning is where the algorithm is trained using the correct target values. In this algorithm, the training is done on the data and error is calculated from the difference between the predicted values and the correct values. Supervised learning can also be further divided into regression and classification algorithms. Regression is where the supervised algorithm is predicting a continuous problem, such as price of a house in house prediction. Classification is where the algorithm tries to predict discrete values for example, 0 or 1, malignant or benign tumor, spam or non-spam emails.

In Machine Learning data attributes are called features. In prediction of housing prices we might use features such as house size, number of bedrooms, number of bathrooms, proximity to downtown etc, all there are features of the algorithm.
Regression algorithms can be used to perform classifications too. One of the most common used algorithms for classification is Logistic Regression. It gives the probabilities of belonging to a given class. Some examples of supervised learning algorithms are:
Linear Regression.
Logistic regression
Neural Networks.
K- Nearest neighbors.
Support Vector Machines.
Decision trees and random forests.

## Unsupervised Learning
Unsupervised learning is a type of algorithm where the prediction is made on data that is unlabeled. In this algorithm, the algorithm is given data and it used patterns to cluster data based on their similarities. Some of the unsupervised learning algorithms are:
Clustering- K Means, Hierarchical Cluster Analysis, Expectation Maximization.
Principal Component Analysis (PCA).
An example of unsupervised learning use case can be seen in Google clustering related news from different sources under one URL link, market segmentation, social network analysis and astronomical data analysis.
One very important task is dimensionality reduction, where we reduce the number of features in our dataset while also keeping the most important features.  This can be done by a combination of some parameters, for example, the square feet of a house is more important than the width of the house. Some factors combined might be more important than one factor, such as age and some illnesses.

## Semi-supervised learning
This is the kind of algorithm where some data is labelled while some data is unlabeled. A great example is when we upload photos on Facebook with the same person the algorithm recognizes them. After applying a label to the person, the algorithm will be able to name all the pictures with the same person. One example of this is the deep belief networks, which is composed of unsupervised Restricted Boltzmann machines stacked on top of each other.
Reinforcement Learning:
Reinforcement learning is a much deeper type of algorithm. In this type of algorithm, the learning system, called an agent, observes the environment, select and perform actions, and get rewards or penalties in the form of negative rewards. The best strategy, called policy defines what action the agent needs to take overtime to maximize rewards. This algorithm can be used in the stock market where the reward for the algorithm is to maximize the returns, and the system figures out what the best actions are to maximize the profits while it is gets negative rewards for losses.

In the programming exercises, that I will be doing, I will be using Octave software. This is the same as Matlab but it is a free open source software. This is because Octave/ Matlab makes it very easy to prototype than using Java/ C++ which takes a lot of code to do. It is therefore advisable to know a prototyping language then after there is proof it works, we can move to the more complicated programming languages such as C++. After proving the code works in Octave, I will move to Python to write my algorithms.
Linear Regression
This is the simplest type of Supervised learning where a model is used to predict a continuous value. The best well known example is the Portland housing price prediction. For the housing price prediction we use a housing price, we will use the housing price dataset.

# **Linear Regression**
Linear regression is a type of supervised learning algorithm where predictions are made using a model that fits the data linearly on the data. Regression comes from the fact that the prediction output values are continuous. Some of the most common notations used in Machine Learning are:
m- Number of training examples.
x- Input variables
y-Output variables
(x,y)- one training example, showing the input and output.
The data that is used to train the algorithm is called the training set.

Size in Ft. squared | Price in dollars ('000)
------------ | -------------
1200 | 720
500 | 320
5000 | 1000

The main objective in Machine Learning is finding a relationship that matches the features in the first column to the output, then use this relation ship to pridict other outcomes given different features. Given the sizes of the houses and proces of the houses, a Supervised Learning algorithm would come up with a hypotheses that fits the featyres to the output in the most accurate way.

We represent the hypotheses as:
\\[h_\theta(x)={\theta_0} + {\theta_1 x}\\] 

Most of you would remember this equation is similar to y = mx + C, which is the equation of a straight line.  This is the simplest type pf algorithm to start with. Later on we will move to algorith equations with more that none feature as well non-linear fuctions. In this equation \\(\theta\\) are called paramteters and x is the feature.

# **Cost function**
From the hypothesis function, we know that different \\(\theta\\) parameters, will yield different linear curves/ hypothesis functions. By using different theta parameters, we are able to fit the curves differenty to different data.

The image below shows equations y = 2, y = 2x and y = 2x + 1.

<img src="https://brianasimba.github.io/MachineLearningblog/images/Theta.png" style="display: block; margin: auto;" />

The objective is to find the best fitting plot given data as shown below:

By using different \\(\theta\\) parameters, we are able to come up with the line the best fits the data. This can be achieved by minimizing the different between the correct values and the hypothesis predictions.


The cost function is defined as:


\\[J(\theta_0, \theta_1) = \frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x{^i})) - y^i)^2\\] 























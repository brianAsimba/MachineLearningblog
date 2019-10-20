---
layout: post
title: Neural Networks Example, Math and code
date: 2019-10-19
excerpt: Neural Networks
tags:
  - Machine Learning
  - Supervised
  - Neural Networks
  - Logistic Regression
  - ReLu
  - Sigmoid 
  - Softmax
comments: true
published: true
---
In this blog post, we will go through the fukk process of forward and backpropagation in Neural Networks. I will debunk the backpropagation mystery that most have accepted to be a black box. After completing the math, I will write code to calculate the same. This will be be particularly be helpful for beginners as they can understand what goes on behind the scenes, especially in backpropagation. Links will be added to assist those who want to dig deeper or want to have a better understanding. The Activation functions that are going to be used are the sigmoid function, Rectified Linear Unit (ReLu) and the Softmax function in the output layer. It is important to note that, it is not important to use different activations as in this example. 

<figure>
<img src="https://brianasimba.github.io/MachineLearningblog/images/Page_1.png" style="display: block; margin: auto;"/>
<figcaption>Branches of Machine Learning</figcaption> 
</figure>


<figure>
<img src="https://brianasimba.github.io/MachineLearningblog/images/Page_2.png" style="display: block; margin: auto;"/>
<figcaption>Branches of Machine Learning</figcaption> 
</figure>

<figure>
<img src="https://brianasimba.github.io/MachineLearningblog/images/3.png" style="display: block; margin: auto;"/>
<figcaption>Branches of Machine Learning</figcaption> 
</figure>

<figure>
<img src="https://brianasimba.github.io/MachineLearningblog/images/Page_4.png" style="display: block; margin: auto;"/>
<figcaption>Branches of Machine Learning</figcaption> 
</figure>

<figure>
<img src="https://brianasimba.github.io/MachineLearningblog/images/Page_5.png" style="display: block; margin: auto;"/>
<figcaption>Branches of Machine Learning</figcaption> 
</figure>

<figure>
<img src="https://brianasimba.github.io/MachineLearningblog/images/Page_6.png" style="display: block; margin: auto;"/>
<figcaption>Branches of Machine Learning</figcaption> 
</figure>


<figure>
<img src="https://brianasimba.github.io/MachineLearningblog/images/Page_7.png" style="display: block; margin: auto;"/>
<figcaption>Branches of Machine Learning</figcaption> 
</figure>


<figure>
<img src="https://brianasimba.github.io/MachineLearningblog/images/Page_8.png" style="display: block; margin: auto;"/>
<figcaption>Branches of Machine Learning</figcaption> 
</figure>


<figure>
<img src="https://brianasimba.github.io/MachineLearningblog/images/Page_9.png" style="display: block; margin: auto;"/>
<figcaption>Branches of Machine Learning</figcaption> 
</figure>



References:
1. https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c

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
In this blog post, we will go through the full process of feedforward and backpropagation in Neural Networks. I will debunk the backpropagation mystery that most have accepted to be a black box. After completing the math, I will write code to calculate the same. This will be be particularly be helpful for beginners as they can understand what goes on behind the scenes, especially in backpropagation. Links will be added to assist those who want to dig deeper or want to have a better understanding. The Activation functions that are going to be used are the sigmoid function, Rectified Linear Unit (ReLu) and the Softmax function in the output layer. It is not mandatory to use different activations functions in each layer as is the case in this example. 

I am looking for questions from the readers and I will be adding notes, as well as links to videos to this blog post as questions come to provide as much clarity as possible. I did the math by hand in order to understand it and be able to explain to anyone who might have any questions. It is beneficial, but not mandatory to have a Calculus background, as it will assist in understanding the Chain rule for differentiation.

<figure>
<img src="https://brianasimba.github.io/MachineLearningblog/images/Page_1.png" style="display: block; margin: auto;"/>
<figcaption>Neural Network Architecture</figcaption> 
</figure>


<figure>
<img src="https://brianasimba.github.io/MachineLearningblog/images/Page_2.png" style="display: block; margin: auto;"/>
<figcaption>First Hidden layer Forward Propagation</figcaption> 
</figure>

<figure>
<img src="https://brianasimba.github.io/MachineLearningblog/images/3.png" style="display: block; margin: auto;"/>
<figcaption>Second Hidden layer Forward Propagation</figcaption> 
</figure>

<figure>
<img src="https://brianasimba.github.io/MachineLearningblog/images/Page_4.png" style="display: block; margin: auto;"/>
<figcaption>Third Hidden layer Forward Propagation</figcaption> 
</figure>

<figure>
<img src="https://brianasimba.github.io/MachineLearningblog/images/Page_5.png" style="display: block; margin: auto;"/>
<figcaption>Activation Functions Derivatives</figcaption> 
</figure>

<figure>
<img src="https://brianasimba.github.io/MachineLearningblog/images/Softmax_derivation.PNG" style="display: block; margin: auto;"/>
<figcaption>Softmax Derivative Explained</figcaption> 
</figure>


<figure>
<img src="https://brianasimba.github.io/MachineLearningblog/images/Page_6.png" style="display: block; margin: auto;"/>
<figcaption>Derivative of Error with Output</figcaption> 
</figure>


<figure>
<img src="https://brianasimba.github.io/MachineLearningblog/images/Page_7.png" style="display: block; margin: auto;"/>
<figcaption>Derivative of the Weight</figcaption> 
</figure>


<figure>
<img src="https://brianasimba.github.io/MachineLearningblog/images/Page_8.png" style="display: block; margin: auto;"/>
<figcaption>Weights Updating for Third Hidden layer</figcaption> 
</figure>


<figure>
<img src="https://brianasimba.github.io/MachineLearningblog/images/page_9A.png" style="display: block; margin: auto;"/>
<figcaption>Back propagation to the Second Hidden layer</figcaption> 
</figure>

<figure>
<img src="https://brianasimba.github.io/MachineLearningblog/images/page10-A.png" style="display: block; margin: auto;"/>
<figcaption>Back propagation to the Second Hidden layer</figcaption> 
</figure>


<figure>
<img src="https://brianasimba.github.io/MachineLearningblog/images/page_11.png" style="display: block; margin: auto;"/>
<figcaption>Back propagation to the Second Hidden layer</figcaption> 
</figure>


<figure>
<img src="https://brianasimba.github.io/MachineLearningblog/images/Page_12.png" style="display: block; margin: auto;"/>
<figcaption>Back propagation to the Second Hidden layer</figcaption> 
</figure>


<figure>
<img src="https://brianasimba.github.io/MachineLearningblog/images/Page_13.png" style="display: block; margin: auto;"/>
<figcaption>Back propagation to the First Hidden layer</figcaption> 
</figure>

<figure>
<img src="https://brianasimba.github.io/MachineLearningblog/images/Page 14.png" style="display: block; margin: auto;"/>
<figcaption>Back propagation to the First Hidden layer</figcaption> 
</figure>


<figure>
<img src="https://brianasimba.github.io/MachineLearningblog/images/Page_15.png" style="display: block; margin: auto;"/>
<figcaption>Back propagation to the First Hidden layer</figcaption> 
</figure>


Here is the github code for this mathematical calculation:

 
<html>  
  
<head> 
    <title> 
        HTML iframe src Attribute 
    </title> 
</head> 
  
<body style="text-align:center;">  
      
   <iframe src="https://brianasimba.github.io/MachineLearningblog/Code for paper.html"
             width="700"></iframe>  
</body>  
  
</html>  











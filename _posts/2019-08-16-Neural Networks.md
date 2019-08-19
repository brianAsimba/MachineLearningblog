---
layout: post
title: Matrices, Multivariate Linear Regression
date: 2019-06-06
excerpt: Neural Networks
tags:
  - Machine Learning
  - Supervised
  - Unsupervised
  - Linear Regression
  - Gradient Descent
comments: true
published: true
---
Birds inspored us to fly, burdock plants inspired velcro, and nature has inspired many other inventions. It seems logical, then ,to look at the brains's architecture for inspiration on how to build an intelligent machine. Neural networks were created to try and mimic the most intelligent machine, the brain. Artificial Neural Networks are versatile, powerful and scalable making them ideal to tackle large and highly complex Machine Learning tasks, suchas classifying billions of images e.g. Google images, powering speech recognition services e.g. Apple's siri, recommending the best videos to watch to hundreds  ot users everyday, e,g. Youtuce, NEtflis, or learning to beat the world champion at the game of Go by examining millions of past game and playing against itself (Deepmeind's AlphaGo).

In the 80s, there was a revival of interest in the ANNs as new network architectures were invented and better training techniques were developed. But in the 1990s, powerful alternative MAchine Learning techniques such as Support  Vectoe Machines were favored by nost researchers as they seemed battere results and stronger theoretial foundations. The question is, whiu is there new revival in ANNs, there are a number of reasons:
1). A huge quantity of data is available to train neural networks and ANNs are outperforming the other ML techniques on very large and complex problems.
2). There is tremendous increase in computation cpower since the 90s which makes it possible to train large neural networks in a reasonable time. This is also in part due to the powerful GPUs.
3). The algorithms have been improved from the 90s algorithms.
4). Some limitations that were theoreticla such as local optima are no longer big issues. MOst of the time the algorithms converge in global optima or local optima that are pretty closr to the global optima.


\\[
\begin{pmatrix} x =
    \frac{1}{5} & \frac{5}{12} & \frac{7}{20} \\\
    \frac{1}{5} & \frac{1}{6} & \frac{1}{10} \\\
    \frac{1}{5} & -\frac{1}{6} & \frac{1}{10}
 \end{pmatrix}  \theta = \begin{pmatrix}
    \frac{1}{5}\\\
    \frac{1}{5}\\\
    \frac{1}{5}
 \end{pmatrix} 
 \\]
 
 Where the hypothesis is given by \\(h_\theta(x)= \frac{1}{1+e^(-\theta x}\\) which is the Sigmoid/Logistic regression activation function. A neuron is modeled by a simple logistic regression. The neural takes the inputs and has the output as shown in the neuron using the activation function to perform the computation. \\(\theta\\) can also be called parameters or weights as it is usually referred to in neural networks.
 Some notations that we are using in the Neural Networks are:
a_{i}^{(j)} = activatioons of the unit "i" in later "j"
\\[\theta^(j)]\\ = matrix of weights/parameters controlling the function mapping from layer j to layer j+1.

\\[a_{1}^{(2)} = g({\theta_{10}}^{(1)} x_0+ {\theta_{11}}^{(1)} x_1 + {\theta_{12}}^{(1)} x_3  +{\theta_{13}}^{(1)} x_3) = g(z_{1}^{(2)}\\]
\\[a_{2}^{(2)} = g({\theta_{20}}^{(1)} x_0+ {\theta_{21}}^{(1)} x_1 + {\theta_{22}}^{(1)} x_3  +{\theta_{23}}^{(1)} x_3) =g(z_{2}^{(2)}\\]
\\[a_{3}^{(2)} = g({\theta_{30}}^{(1)} x_0+ {\theta_{31}}^{(1)} x_1 + {\theta_{32}}^{(1)} x_3  +{\theta_{33}}^{(1)} x_3) =g(z_{3}^{(2)}\\]
\\[h_{\theta}{(x)} = a_{1}^{(3)} = g({\theta_{10}}^{(2)} x_0+ {\theta_{11}}^{(2)} x_1 + {\theta_{12}}^{(2)} x_3  +{\theta_{13}}^{(2)} x_3)\\]

From the neural network, we cna see that if the network has s_j units in layer j, and s_j +1 units in layer j+1, then \\(\theta^(j)\\) will be dimension s_(j+1) * (s_j +1). The dimesnion of the matrix of the previous layer is s_j+1 because there is a bias unit added to the previous layer.

If we vectorize the expressions, we get:
\\[z_^{(2)} = {\theta_}^{(1)} x\\]
\\[a_^{(2)} = {g(z^{(2)})\\], where g is the activation function that applied the multiplication elementwise.
Wwe then add the \\(a_{0}^{(2)} = 1\\) which is the bias term, making it a 4 dimensional vector.
Then, finally we can calculate\\(z_^{(3)} = {\theta_}^{(2)} a^(2)\\) and (h_{\theta}{(x)} = a^{3} ={g(z^{(3)})\\)

Forward propagation is the propagation of the inputs via activation functions to the hidden layers and finally to the output layer.As I states previously on my first post, using vectorized format reduces computation time as it eliminated using for loops in the code.



The diagram below shows the Neural network. 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

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

Some examples of simple neural networks are these:
For the Login x1 and x2, where x1 and x2 \\(\epsilon)\\ {0,1}, here is the Neural Network:
y = x_1 and x_2



From the Neural Network, we see that:
\\[h_\theta(x)=g(-30+20x_1+20x_2)\\] 
Creating a truth table for this yields the following
x_1    x_2 | \\[h_\theta(x)]\\
0    0     | g(-30)~ 0
0    1     | g(-10)~ 0
1    0     | g(-10)~ 0
1    1     | g(10) ~ 1

From the truth table, you can see that the parameters x_1 amd x_2, gives the value 1 only when both x_1 and x_2 are positive.


 
For the logic x1 or x2, where x1 and x2 \\(\epsilon)\\ {0,1}, here is the Neural Network:
y = x_1 or x_2



From the Neural Network, we see that:
\\[h_\theta(x)=g(-10+20x_1+20x_2)\\] 
Creating a truth table for this yields the following
x_1    x_2 | \\[h_\theta(x)]\\
0    0     | g(-10)~ 0
0    1     | g(10)~ 1
1    0     | g(10)~ 1
1    1     | g(10)~ 1

From the truth table, you can see that the parameters x_1 or x_2, gives the value 1 only when either x_1 or x_2 is positive. 
 
For the logic NOT x1 and x2, where x1 and x2 \\(\epsilon)\\ {0,1}, here is the Neural Network:
y = NOT x_1 AND x_2



From the Neural Network, we see that:
\\[h_\theta(x)=g(10-20x_1-20x_2)\\] 
Creating a truth table for this yields the following
x_1    x_2 | \\[h_\theta(x)]\\
0    0     | g(10)~ 1
0    1     | g(-10)~ 0
1    0     | g(-10)~ 0
1    1     | g(-30)~ 0

From the truth table, you can see that the parameters x_1 or x_2, gives the value 1 only when either x_1 or x_2 is positive.  


For the logic XNOR x1 XNOR x2, where x1 and x2 \\(\epsilon)\\ {0,1}, here is the Neural Network:
y = NOT x_1 XNOR x_2



THis is a Neural Network that combines the AND, OR and AND NOT functions to have even more complex classifications.
Creating a truth table for this yields the following
x_1    x_2 | a_{1}^{(2)}   a_{2}^{(2)}  |\\[[h_\theta(x)]\\
0    0     |      0              1      |       1
0    1     |      0              0      |       0
1    0     |      0              0      |       0
1    1     |      0              0      |       1

From the truth table, you can see that the parameters x_1 or x_2, gives the value 1 only when either x_1 or x_2 is positive.

When working with Neural Networks, it is always good to print out the sizes of the matrices and vectors to ensure that the vectors can be multiplied together to get the correcet sizes. We are going to look into a code example where we are going to be running al algorithms for multi-class classification.

First, we will bring in the data which for this eample has 5000 training examples in which each training example is a 20 pixel by 20 pixel grayscale image of the digit.The 20 by 20 pixedls is "unrolled" into a 400 dimensinal vector. THis will result in a 5000 by 400 matrix X where each row is a training example as shown below:

 \begin{pmatrix}
   - (x^(1))^T - \\\
   - (x^(2))^T - \\\
   - (x^(3))^T - \\\  
   \vdots\\\
   - (x^(m))^T -  
  \end{pmatrix} 
 
 The y is a 5000-dimensional array that contains the label for the training set. We will beusing multiple one-vs-all logistic regression models to build the multi-class classifier. Next, we will vectorize the code, which will enable it to be computationally efficient. For the 10 classes that we want to recognize the digits, we will need to train 10 separate logistic regression classifiers. 

From Logistic Regression, these are the Equations  for the cost function and the hypothesis:
\\[ J(\theta)_ = \frac{1}{m}\sum_{i=1}^{m}[-y log(h\theta(x{^i})) - (1-y({^i})) log (1-h\theta(x{^i}))]\\] 
 
We, then compute each element in the summation, we compute \\(h\theta(x{^i})\\) for evey training example,i. where  \\(h\theta(x{^i}) = g(\theta^T x^(i))\\) and \\(g(z) = frac{1}(1+e^(-1)}\\) is the sigmoid function. X and \theta are defined as:
 \begin{pmatrix} X =
   - (x^(1))^T - \\\
   - (x^(2))^T - \\\
   - (x^(3))^T - \\\  
   \vdots\\\
   - (x^(m))^T -  
  \end{pmatrix}  \and
  \begin{pmatrix} \theta =
   \theta_(0))^T - \\\
   \theta_(1))^T - \\\  
   \vdots\\\
   \theta_n 
  \end{pmatrix} 
 Then by computingmatrix product X\theta, we have:
  \\[
\begin{pmatrix}
     - (x^(1))^T\theta -\\\
     - (x^(2)^T\theta -\\\
     \vdots\\\
     - (x^(m))^T\theta -      
 \end{pmatrix}  \= \begin{pmatrix}
    - \theta^T(x^(1)) -\\
     - \theta^T(x^(2) -\\\
     \vdots\\\
     - \theta^T (x^(m))   
  \end{pmatrix}  \\] 
 
 
 These equations provides us the means to calculate all the products \\(\theta^T x^(i)\\) in one matrix calculaion.The gradient for the unregularized logistic regression is given by:

\\[\frac{\partial J}{\partial \theta_j} =  \frac{1}{m}\sum_{i=1}^{m}(h_\theta(x{^i})) - y^(i))(x_j{^(i)})\\] 

In order to vectorize, we should first write out the partial derivatives as shown:

 \\[
\begin{pmatrix}
     \frac{\partial J }{partial\theta_0} \\\
     \frac{\partial J }{partial\theta_1}  \\\
     \frac{\partial J }{partial\theta_2}  \\\
     \vdots\\\
     \frac{\partial J }{partial\theta_n}     
 \end{pmatrix}  \=  \frac{1}{m}\begin{pmatrix} 
 \begin{pmatrix}
     \sum_{i=1}^{m}[(h_\theta(x{^i})) - y({^i}))x_0{^i}) \\\
     \sum_{i=1}^{m}[(h_\theta(x{^i})) - y({^i}))x_1{^i})  \\\
     \sum_{i=1}^{m}[(h_\theta(x{^i})) - y({^i}))x_2{^i})  \\\
     \vdots\\\
     \sum_{i=1}^{m}[(h\theta(x{^i})) - y({^i}))x_n{^i})    
 \end{pmatrix}\\] 
  
 The vectorized version yields:
\\[\frac{1}{m}\sum_{i=1}^{m}[(h\theta(x{^i})) - y({^i}))x^{i}) =\frac{1}{m}X^T [(h\theta(x) - y)\\] 
 
 where:
 \\[  \begin{pmatrix} (h_\theta(x) - y) \=
     (h_\theta(x{^1})) - y({^1}) \\\
     (h_\theta(x{^2})) - y({^2})  \\\
     \vdots\\\
     (h_\theta(x{^m})) - y({^m})  \\\
 \end{pmatrix}  
\\] 
 
 ## Vectorizing Regularized Logistic Regression
Now we will add regularization to the Logistic Regression in order to avoid overfitting of the parameters. The Regularized Logistic regression is given by:
\\[ J(\theta)_ = \frac{1}{m}\sum_{i=1}^{m}(-y log(h\theta(x{^i})) - (1-y({^i})) log (1-h\theta(x{^i}))) + \frac{\lambda}{2m}\sum_{j=1}^{n} \theta_j{^2}\\] 
Where we do not regalarize the \\(\theta_0\\) whioch is used for the bias term. The partial derivatives for the regalarized logistic regressions is given as follows:

\\[\frac{\partial J}{\partial \theta_j} =  \frac{1}{m}\sum_{i=1}^{m}(h_\theta(x{^i})) - y^(i))(x_j{^(i)})\\] for j=0
\\[\frac{\partial J}{\partial \theta_j} =  \frac{1}{m}\sum_{i=1}^{m}(h_\theta(x{^i})) - y^(i))(x_j{^(i)}) +\frac{\lambda}{m}\theta_j\\] 

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

---
layout: post
title: Matrices, Multivariate Linear Regression
date: 2019-06-06
excerpt: Supervised Machine Learning
tags:
  - Machine Learning
  - Supervised
  - Unsupervised
  - Linear Regression
  - Gradient Descent
comments: true
published: true
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

## Matrices and Vectors:
Matrices and vectors are arrays that are used to express any kinds of data processing tasks concisely that might otherwise require writing loops. Practice of replacing loops with matrices an vectors is referred to as vectorization. Vectorization will often be one or more orders of magnitude faster than their pure equivalents using loops. One of the powerful methods that makes vectorization very effective is broadcasting, which we will discuss about later on. First, let us start with some matrix naming:

$$
\left( \begin{array}{cc}
      y_1 & 2 \\\\
      \vdots & 6 \\\\
      y_n & 5
    \end{array} \right)
$$




Matrices are named by number of rows x number of columns. The first matrix is a 3X2 while the second one is 2X3 matrix. We can now refer to the elements of the matrix. The elements are named using i and j, where i is the row and j is the column of the element in the matrix. In matrix A and B:

\\[A_{3,2} = 10\\]
\\[A_{1,2} = 3\\]
\\[B_{2,2} = 9\\]



  

## Vector
A vector in a matrix with 1 column. An example of a vector is:

$$
\left( \begin{array}{c}
      120 \\
      100 \\
      50
    \end{array} \right)
$$




M = \left( \begin{array}{ccc}
a & C & c \\
b & a & a \\
d & a & v \\
\end(array} \right)


This is shown as 

$$
\begin{align*}
  & \phi(x,y) = \phi \left(\sum_{i=1}^n x_ie_i, \sum_{j=1}^n y_je_j \right)
  = \sum_{i=1}^n \sum_{j=1}^n x_i y_j \phi(e_i, e_j) = \\
  & (x_1, \ldots, x_n) \left( \begin{array}{ccc}
      \phi(e_1, e_1) & \cdots & \phi(e_1, e_n) \\
      \vdots & \ddots & \vdots \\
      \phi(e_n, e_1) & \cdots & \phi(e_n, e_n)
    \end{array} \right)
  \left( \begin{array}{c}
      y_1 \\
      \vdots \\
      y_n
    \end{array} \right)
\end{align*}
$$





$$\begin{eqnarray} 
x' &=& &x \sin\phi &+& z \cos\phi \\\\
z' &=& - &x \cos\phi &+& z \sin\phi \\\\
\end{eqnarray}$$


$$
\begin{array}{ccc}
y & 3 \\\
y & 2 \\\
z & 3
\end{array}
$$

\begin{pmatrix}
    \frac{1}{5} & \frac{5}{12} & \frac{7}{20} \\\
    \frac{1}{5} & \frac{1}{6} & \frac{1}{10} \\\
    \frac{1}{5} & -\frac{1}{6} & \frac{1}{10}
 \end{pmatrix}

Vectors elements are referred to using the syntax: [/y_i = i^{th}\\] element. For example, in the vector B, y_1 = 120, y_2 = 100.

## Matrix Addition
We add matrices by using the syntax below:
\\[
\begin{pmatrix}
    \frac{1}{5} & \frac{5}{12} & \frac{7}{20} \\\
    \frac{1}{5} & \frac{1}{6} & \frac{1}{10} \\\
    \frac{1}{5} & -\frac{1}{6} & \frac{1}{10}
 \end{pmatrix}  \+ \begin{pmatrix}
    \frac{1}{5} & \frac{5}{12} & \frac{7}{20} \\\
    \frac{1}{5} & \frac{1}{6} & \frac{1}{10} \\\
    \frac{1}{5} & -\frac{1}{6} & \frac{1}{10}
 \end{pmatrix} \=\begin{pmatrix}
    \frac{1}{5} & \frac{5}{12} & \frac{7}{20} \\\
    \frac{1}{5} & \frac{1}{6} & \frac{1}{10} \\\
    \frac{1}{5} & -\frac{1}{6} & \frac{1}{10}
 \end{pmatrix}
 \\]
 Scalar multiplication is performed as below:
 \\[3 \times
\begin{pmatrix}
    \frac{1}{5} & \frac{5}{12} & \frac{7}{20} \\\
    \frac{1}{5} & \frac{1}{6} & \frac{1}{10} \\\
    \frac{1}{5} & -\frac{1}{6} & \frac{1}{10}
 \end{pmatrix}
 \\]

\\[
\begin{pmatrix}
    \frac{1}{5} & \frac{5}{12} & \frac{7}{20} \\\
    \frac{1}{5} & \frac{1}{6} & \frac{1}{10} \\\
    \frac{1}{5} & -\frac{1}{6} & \frac{1}{10}
 \end{pmatrix} /3
 \\]
 
 Matrix-vector multiplication:
 \\[
\begin{pmatrix}
    \frac{1}{5} & \frac{5}{12} & \frac{7}{20} \\\
    \frac{1}{5} & \frac{1}{6} & \frac{1}{10} \\\
    \frac{1}{5} & -\frac{1}{6} & \frac{1}{10}
 \end{pmatrix}  \times \begin{pmatrix}
    \frac{1}{5}\\\
    \frac{1}{5}\\\
    \frac{1}{5}
 \end{pmatrix} 
 \\]
 
 Here is a simple example of how matrices are used in Machine Learning in the House prediction algorithm. If we have four houses and we want to computethe hupoithesis, here is how we would go about it.
 
 House sizes:
 1500
 1300
 2200
 2500
 
The hypothesis we want to use is \\(h_\theta(x)=50 + {0.5 x}\\) , then this can be expressed as a matrix or vector as:
\\[
\begin{pmatrix}
    1 & 1500 \\\
    1 & 1300 \\\
    1 & 2200 \\\
    1 & 2500
 \end{pmatrix}  \times \begin{pmatrix}
    50 \\\
    0.5
 \end{pmatrix} 
 \\]
The same results can be obtained using this hyoithesis, but using For loop. However, using for loops is not effective and it is much more computationally costly than using Matrices.
 
Matrix-Matrix Multiplication:
A matrix of dimension mxn can only be multiplied by another matrix of dimension nxo.
 
 \\[
\begin{pmatrix}
    \frac{1}{5} & \frac{5}{12} & \frac{7}{20} \\\
    \frac{1}{5} & \frac{1}{6} & \frac{1}{10} \\\
    \frac{1}{5} & -\frac{1}{6} & \frac{1}{10}
 \end{pmatrix}  \times \begin{pmatrix}
    \frac{1}{5}\\\
    \frac{1}{5}\\\
    \frac{1}{5}
 \end{pmatrix} 
 \\]
 
 One area where Matrix-Matrix multiplication is used is when computing different hypotheses outputs. An example is this one:


Size in Ft. squared | 
------------ | -------------
1200 | 
500 | 
5000 | 
 
And we have four competing hyopothesis:
\\[h_\theta(x)=100 + {0.75 x}\\]
\\[h_\theta(x)=30 + {0.35 x}\\]
\\[h_\theta(x)=20 + {0.55 x}\\]
\\[h_\theta(x)=80 + {0.5 x}\\]  
 
\\[
\begin{pmatrix}
    \frac{1}{5} & \frac{5}{12} & \frac{7}{20} \\\
    \frac{1}{5} & \frac{1}{6} & \frac{1}{10} \\\
    \frac{1}{5} & -\frac{1}{6} & \frac{1}{10}
 \end{pmatrix}  \times \begin{pmatrix}
    \frac{1}{5}\\\
    \frac{1}{5}\\\
    \frac{1}{5} \end{pmatrix} \= \begin{pmatrix}
    \frac{1}{5} & \frac{5}{12} & \frac{7}{20} \\\
    \frac{1}{5} & \frac{1}{6} & \frac{1}{10} \\\
    \frac{1}{5} & -\frac{1}{6} & \frac{1}{10}
 \end{pmatrix}\\] 
 
The final matrix is the predictions from the hypotheses, with the first column is the predictions of the first hypothesis and so on and so forth.

## Inverse and Transpose
Inverse in Mathematics, is the number or array that you can multiply a Real number with to get the identity number. In scalars, identity is the number 1, meaning any real number multiplied by 1 yield that Real number as the answer. For example \\(x + 1 = 1 + x\\). In Matrices, Identity is denoted by \\(I_{n x n}\\). Identity matrix for Matrices have 1 in the diagonal and 1 everywhere else in the Matrix as shown below:

\\[
\begin{pmatrix}
    1 & 0 & 0 \\\
    0 & 1 & 0 \\\
    0 & 0 & 1
 \end{pmatrix}
 \\]


\\[
\begin{pmatrix}
    1 & 0 & 0 & 0\\\
    0 & 1 & 0 & 0\\\
    0 & 0 & 1 & 0\\\
    0 & 0 & 0 & 1
 \end{pmatrix}
 \\]
For Matrices, multiplying the matrices by the Identity matrix yields the Identity Matrix. For example:
\\[
\begin{pmatrix}
    3 & 2 & 1 \\\
    5 & 3 & 2 \\\
    1 & 5 & 3
 \end{pmatrix} \times \begin{pmatrix}
    \frac{1}{7} & \frac{1}{7} & \frac{-1}{7}\\\
    \frac{13}{7} & \frac{-8}{7} & \frac{1}{7}\\\
    \frac{-22}{7} & \frac{13}{7} & \frac{1}{7}
 \end{pmatrix} \= \begin{pmatrix}
    1 & 0 & 0\\\
    0 & 1 & 0\\\
    0 & 0 & 1
 \end{pmatrix}
 \\]


Transpose in Matrices is the flipping of a Matrix so that the columns become the rows as shown below:

\\[B=
\begin{pmatrix}
    3 & 2 & 1 \\\
    5 & 3 & 2 \\\
    1 & 5 & 3
 \\]

\\[B^{T}=
\begin{pmatrix}
    3 & 2 & 1 \\\
    5 & 3 & 2 \\\
    1 & 5 & 3
 \\]

## Multi-variate Linear Regression

We have gone through Linear Regression wiht a single variable in the first blog post. We now have an idea of how Matrices should be used in Machine Learning to make coding computationally efficient. Multi-variate Linear Regression is where the hypothesis has more than one variable. An example of this can be seen in the House Price prediction. Let us say we have the following data:


Size in Ft. squared | Number of Bedrooms | Number of rooms | Proximity from Downtown\\(Ft^2\\)
------------ | -------------
1200 | 2 | 10 | 500
500 | 3 | 12 | 50
5000 | 5 | 15 | 1000
1000 | 3 | 10 | 5000
... | ... | ... | ...

When using Multiple feature, we would name the features above as \\(X_1\\),\\(X_2\\),\\(X_3\\),\\(X_4\\) for the Size in Ft., Number of Bedrooms, Number of Rooms and Proximity to Downtown respectively. m willbe the number of training examples as before and n as the number of imput feature, \\(x^{(2)}\\) will denote the \\(i^{th}\\) training eample in the training set, while \\(x_{j}^{i}\\) will the \\(j^{th}) value in the \\(i^{th}) tranining example.

For example, in the housing data above:

\\[x^{(4)} =
\begin{pmatrix}
    1 & 0 & 0 & 0\\\
    0 & 1 & 0 & 0\\\
    0 & 0 & 1 & 0\\\
    0 & 0 & 0 & 1
 \end{pmatrix}
 \\]
 
 Where \\(x_{1}^{4}\\) = 1000, \\(x_{2}^{4}\\) = 3 etcetera.
 
 Our hyopothesis will now bw modified to include the extra parameters as shown below:
 
 \\[h_\theta(x)={\theta_0 X_0} + {\theta_1 X_1}+... +\{\theta_n X_n}\\] 
 

\\[h_\theta(x)={\theta^{T} X\\] 

Where:

\\[X =
\begin{pmatrix}
    X_0\\\
    X_1\\\
    X_2\\\
    \vdots\\\
    X_n
 \end{pmatrix}
 \begin{pmatrix}
    \theta_0\\\
    \theta_1\\\
    \theta_2\\\
    \vdots\\\
    \theta_n
 \end{pmatrix} 
 
 \\]

For multiple variables, these are the equations of the Hypothesis, Parameters, cost function and gradient descent:
Hypothesis: \\(h_\theta(x)= {\theta^{T} X = {\theta_0}x_0 + {\theta_1 x_1} + {\theta_2 x_2} + ... + {\theta_n x_n}\\)
Parameters: \\({\theta\\)
Cost function: \\(J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x{^i})) - y^i)^2\\)
Gradient Descent: Repeat{
\\[\theta_j  :=\theta_j -\alpha\frac{\partial}{\partial \theta_j} J(\theta)\\]
}        (simulateneously update for every  j = 0,1,2,...n)

For multiple parameters, update the \\(\theta\\) parameters as shown below:
Repeat{
\\[  \theta_j := \theta_j  - \alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x{^i})) - y^i)x_{j}^{i}\\]
        (simulateneously update for every  j = 0,1,2,...n)
        }
  
From this update algorithm, if we had 3 parameters, they will be updated as following:
\\[  \theta_0 := \theta_0  - \alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x{^i})) - y^i)x_{0}^{i}\\]
\\[  \theta_1 := \theta_1  - \alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x{^i})) - y^i)x_{1}^{i}\\]
\\[  \theta_2 := \theta_2  - \alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x{^i})) - y^i)x_{2}^{i}\\]
\\[  \theta_3 := \theta_3  - \alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x{^i})) - y^i)x_{3}^{i}\\]

## Feature Scaling
Feature scaling is the practivce of makling sure that all the features are taking the same scale. For example, in the housing analysis, the bedrooms and the size of the houses are of different scales. Trying to solve an algorithm with the different feature scales will prove to be problematic and might take a long time to converge. Scaling will enable us to shrink the features to the same scale, preferably in the range \\(-1\leq x_i\leq 1\\) or as close as possible to this range. For example house sizes \\(x_1\\) ranging from 1-10000 \\(ft^2\\), and \\(x_2\\), number of bedroom ranging from 1-5 would be scaled as:
\\[x_1 = \frac{size}{10000}]
\\[x_1 = \frac{bedrooms}{6}]
One way that features are scaled is using Mean Normalization, which uses this formula:
\\[\frac {x_i - \mu_i}{S_1}\\], where \\(\mu\\) is the average and \\(S_1\\) is the range. Note that we do not apply to \\(x_0\\) since it has a value of 1.
Applying the mean normalization to the houses would yield the following:
\\[x_1 = \frac{size - 5000}{10000}\\], where 5000 is the average size of the houses.

##Learning Rate
In Gradient Descent, the idea if to tweak parameters iteratively to minimize the cost function.  good analogy of how gradient descent works is to imagine that you are on top of a mountain and trying to ge do the bottom of the mountain. The quickest way to get to the bottom is to use the steepest slope to the bottom of the mountain. What Gradient Descent does it to measure the local slope of the error function with regards to the parameter \\(theta\\) and it moves in the direction that will minimize the slope.This is done iteratively until the gradient of 0 is achieved.
An important parameter in Gradient Descent is the Learning Rate hyperparameter. It determines what stepp size the algorithm takes to reach to the global minima. If it is set to be too small, the algorithm converges very slowly which can be computationally costly, while if the Learning Rate is too large the algorithm will diverge instead of converging. One issue with Gradient Descent is that it can be stuck in the local optimum instead if a global optimum. This is an issue in algorithms that are not linear, such as Linear Regression. This is an issue we will deal with when I cover other typees of algorithms.

A sanity check to make sure that gradient descent is running properly is to plot a graph of the Cost Function \\(J_\theta) against the number of iterations. If properly coded the cost should be increasing with every iteration. The number of iterations that it takes to solve al algorithm varies from algorithm to algorithm. The best way to come up with the best \\(\alpha)\\ is to run different \\(alpha)\\ values and to test which one gives the convergence after the number of iterations require.

## Polynomial regression    
Polynomial regression is where the hypothesis used for the machine learning algorithm uses a polyunomial function such as a square functin, quadratic function, square root function. Using these features can assist in matching the data more accurately.

## Normal Equation
Normal equation is the way to solve for the \\(\theta)\\ parameters analytically using mathematical equations to find a closed-form colution.

If we assume that we have a cost function of:
\\[J(\theta)=a(\theta)^2 + b(\theta) + c\\]

Minimizing a function in Calculus, required that we take the derivative of the function with respect to \\(\theta)\\ and set it equal to 0 to solve for the \\(\theta)\\ values.

Therefore the equation above would be as shown below:

\\[\frac{\partial}{\partial \theta} J(\theta_0) = 0 ;;; set to 0 (for every j)\\]
The solve for \\(\theta\\)

Since we now have mutpiple \\(\theta\\) parameters, we have the cost function changed to:

\\[J(\theta_0, \theta_1, \theta_2...,\theta_m) = \frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x{^i})) - y^i)^2\\] 
Then we solve for \\(\theta\\) parameters from this differentiation.

Suppose we are working with the data for the housing prices:
Size in Ft. squared | Number of Bedrooms | Number of rooms | Proximity from Downtown\\(Ft^2\\) | Prices($)
------------ | -------------
x_1 | x_2 | x_3 | x_4 | y
1200 | 2 | 10 | 500 | 500000
500 | 3 | 12 | 50 | 200000
5000 | 5 | 15 | 1000 | 300000
1000 | 3 | 10 | 5000 | 250000
3000 | 2 | 15 | 50000 | 150000
... | ... | ... | ... | ...

To use the Normal Equation, we first construct a Matrix of this data, we first add a column for x_1 and consruct the Matrix as shown

 | Size in Ft. squared | Number of Bedrooms | Number of rooms | Proximity from Downtown\\(Ft^2\\) | Prices($)
------------ | -------------
x_0 |x_1 | x_2 | x_3 | x_4 | y
1 | 1200 | 2 | 10 | 500 | 500000
1 | 500 | 3 | 12 | 50 | 200000
1 | 5000 | 5 | 15 | 1000 | 300000
1 | 1000 | 3 | 10 | 5000 | 250000
1 | 3000 | 2 | 15 | 50000 | 150000
... | ... | ... | ... | ... | ...

Then the Matrix will be written as:
\\begin{pmatrix}
    1 & 1200 & 2 & 10 & 500 & 500000\\\
    1 & 500 & 3 & 12 & 50 & 200000\\\
    1 & 5000 & 5 & 15 & 1000 & 300000\\\
    1 & 1000 & 3 & 10 & 5000 & 250000\\\
    1 & 3000 & 2 & 15 & 50000 & 150000
 \end{pmatrix}
 \begin{pmatrix}
 y =
    500000\\\
    200000\\\
    300000\\\
    250000\\\
    150000
 \end{pmatrix} 
 
 \\]

\\(\theta\\) is then solved using the equation:
[\\(\theta) = (X^{T}X)^{-1}(X^{T}Y)]\\

A question that is normally asked is when to use the Noarmal Equation versus Gradient Descent. Here are some advantages and disadventages of both methods:
For Gradient Descent, the advantages and disadvantages are:
Merits | Demerits
Need (\alpha)\\ for Gradient Descent steps | Can be used to solve when there are a lot of features, more than 10^6.
Needs a lot iterations to converge to a solution

Normal Equation, has the following merits and demerits:
Merits | Demerits
There is no need for (\alpha)\\ since it is solved analytically. | Slow when there are lots of features, more than 10^6.
No need to invert a large matrix as requires in Normal Equation. | Inverting a large matrix is computationally expensive.

Point to note is that the Normal Equations only works for Linear Regression that do not have a lot of features. In complex algorithms such as Neural Networks, the Normal Equation does not work. Therefore, the Gradient Descent is still very useful to solve complex problems.

## Linear Regression with Single variable
We will begin our first Machine Learning algorithm with the Linear Regression algorithm. It is always good practice, to plot the data to view that data that we are working with. This wil be done by plotting the points in a scatter plot as shown below showing scatter plot between the profit and population.

<figure>
<img src="https://brianasimba.github.io/MachineLearningblog/images/Screen Shot 2019-07-27 at 5.22.16 PM" style="display: block; margin: auto;"/>
<figcaption>Branches of Machine Learning</figcaption> 
</figure>

We can plot this because it is only one dimensional plot with one variable. Later on, when we work with multi-dimensional problems where we cannot plot 2 dimensinal plots. Now, we need to fit a linear regression line to the dataset using Gradient Descent.
The objective of this model is to minimize the cost function:
\\[J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x{^i})) - y^i)^2\\]  

where the hypothesis (\h_(theta(x))\\ is given by the linear model:
\\[J(\theta)(x) = \(\theta^T)x =_(\theta(0)) + (\(theta_1))x_1\\]

We are going to then adjust the (\theta_j) parameters to minimize the cost function. 

\\[\theta_j =  \theta_j -\alpha \frac{1}{m}\sum_{i=1}^{m}(h_\theta(x{^i})) - y^i)x_j^(i)\\]  (simulteneously update 
\(theta_j\\) for all j).
Where the (\theta\\) parameters are adjusted in each iteration. NOte that we add an additional column of ones in the X matrix to take into account the intercept,.

<figure>
<img src="https://brianasimba.github.io/MachineLearningblog/images/Linear fit.png" style="display: block; margin: auto;"/>
<figcaption>Branches of Machine Learning</figcaption> 
</figure> 









<figure>
<img src="https://brianasimba.github.io/MachineLearningblog/images/Gradient Descent.png" style="display: block; margin: auto;"/>
<figcaption>Branches of Machine Learning</figcaption> 
</figure> 

From the graph, you can see how \\(J_(\theta)\\) varies with change in (\theta_0)\\ and (\theta_1)\\. The cost function is bow-shaped and has a global minimum. The center point is the optimal point for (\theta_0)\\ and (\theta_1)\\. 

Debugging:
1). If you are having errors while running, inspect the matris dimensions to make sure you are adding and multiplying the correct dimunsions of matrices.
2). Octave interprets mathj operators to be matrix operations. THis can cause incompatibility errors. To perform normal multiplication of matrices, we need to add a "dot". A*B performs matrix multiplication while A.*B performs element-wise multiplication.
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 



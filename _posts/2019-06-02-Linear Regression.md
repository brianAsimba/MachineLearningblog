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
 
 Our hyoithesis will now bw modified to include the extra parameters as shown below:
 
 \\[h_\theta(x)=50 + {0.5 x}\\]



































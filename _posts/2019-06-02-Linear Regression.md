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


\begin{matrix}
 2 &  3 \\
 10 & 8 \\ 
 5 &  10   
 \end{matrix}



Matrices are named by number of rows x number of columns. The first matrix is a 3X2 while the second one is 2X3 matrix. We can now refer to the elements of the matrix. The elements are named using i and j, where i is the row and j is the column of the element in the matrix. In matrix A and B:

\\[A_{3,2} = 10\\]
\\[A_{1,2} = 3\\]
\\[B_{2,2} = 9\\]


\begin{equation}
   \begin{matrix} 
   a_{11} & a_{12} & a_{13}  \\
   a_{21} & a_{22} & a_{23}  \\
   a_{31} & a_{32} & a_{33}  \\
   \end{matrix} 
\end{equation}


\begin{array}{ccc}
      \phi(e_1, e_1) & \cdots & \phi(e_1, e_n) \\
      \vdots & \ddots & \vdots \\
      \phi(e_n, e_1) & \cdots & \phi(e_n, e_n)
    \end{array} \right)

  

## Vector
A vector in a matrix with 1 column. An example of a vector is:

$$
\left( \begin{array}{c}
      y_1 \\
      \vdots \\
      y_n
    \end{array} \right)
$$




M = \left( \begin{array}{ccc}
a & C & c \\
b & a & a \\
d & a & v \\
\end(array} \right)


This is shown as $$\begin{matrix} 1 & x & x^2 \\1 & y & y^2 \\1 & z & z^2 \\ \end{matrix}$$

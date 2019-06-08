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

## Matrices and Vectors:
Matrices and vectors are arrays that are used to express any kinds of data processing tasks concisely that might otherwise require writing loops. Practice of replacing loops with matrices an vectors is referred to as vectorization. Vectorization will often be one or more orders of magnitude faster than their pure equivalents using loops. One of the powerful methods that makes vectorization very effective is broadcasting, which we will discuss about later on. First, let us start with some matrix naming:

A = \begin{bmatrix}
 2&  3&  \\ 
 10&  8&  \\ 
 5&  2&   
\end{bmatrix}

B= \begin{bmatrix}
 2&  3&  2& \\ 
 1&  9&  6& 
\end{bmatrix}
Matrices are names by number of rows x number of columns. The first matrix is a 3X2 while the second one is 2X3 matrix. We can now refer to the elements of the matrix. The elements are named using i and j, where i is the row and j is the column of the element in the matrix. In matrix A:
A_{{1},{1}} = 2
A_{{3},{1}} = 5


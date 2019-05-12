+++
title = "Brief Introduction of Label Propagation Algorithm"
author = ["KK"]
date = 2017-07-16T21:45:00+08:00
lastmod = 2019-05-12T21:21:51+08:00
tags = ["Machine Learning", "Label Propagation"]
draft = false
noauthor = true
nocomment = true
nodate = true
nopaging = true
noread = true
+++

As I said before, I'm working on a text classification project. I use `doc2vec` to convert text into vectors, then I use LPA to classify the vectors.

LPA is a simple, effective semi-supervised algorithm. It can use the density of unlabeled data to find a hyperplane to split the data.

Here are the main stop of the algorithm:

1.  Let $ (x\_1,y1)...(x\_l,y\_l)$ be labeled data, $Y\_L = \\{y\_1...y\_l\\} $ are the class labels. Let \\((x\_{l+1},y\_{l+u})\\) be unlabeled data where \\(Y\_U = \\{y\_{l+1}...y\_{l+u}\\}\\) are unobserved, usually \\(l \ll u\\). Let \\(X=\\{x\_1...x\_{l+u}\\}\\) where \\(x\_i\in R^D\\). The problem is to estimate \\(Y\_U\\) for \\(X\\) and \\(Y\_L\\).
2.  Calculate the similarity of the data points. The most simple metric is Euclidean distance. Use a parameter \\(\sigma\\) to control the weights.

\\[w\_{ij}= exp(-\frac{d^2\_{ij}}{\sigma^2})=exp(-\frac{\sum^D\_{d=1}{(x^d\_i-x^d\_j})^2}{\sigma^2})\\]

Larger weight allow labels to travel through easier.

1.  Define a \\((l+u)\*(l+u)\\) probabilistic transition matrix \\(T\\)

\\[T\_{ij}=P(j \rightarrow i)=\frac{w\_{ij}}{\sum^{l+u}\_{k=1}w\_{kj}}\\]

\\(T\_{ij}\\) is the probability to jump from node \\(j\\) to \\(i\\). If there are \\(C\\) classes, we can define a \\((l+u)\*C\\) label matrix \\(Y\\), to represent the probability of a label belong to class \\(c\\). The initialization of unlabeled data points is not important.

1.  Propagate \\(Y \leftarrow TY\\)
2.  Row-normalize Y.
3.  Reset labeled data's Y. Repeat 3 until Y converges.

In short, let the nearest label has larger weight, then calculate each label's new label, reset labeled data's label, repeat.

{{< figure src="/images/label_spreading.png" width="400" >}}

Ref:

1.  [Learning from Labeled and Unlabeled Data with Label Propagation](http://mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf)
2.  [标签传播算法（Label Propagation）及Python实现](http://blog.csdn.net/zouxy09/article/details/49105265)

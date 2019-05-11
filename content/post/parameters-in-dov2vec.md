+++
title = "Architectures"
author = ["KK"]
lastmod = 2019-05-11T22:36:00+08:00
tags = ["machine learning", "word2vec"]
draft = false
noauthor = true
nocomment = true
nodate = true
nopaging = true
noread = true
+++

## Hierarchical Softmax {#hierarchical-softmax}

Encode words into a huffman tree, then each word has a Huffman code. The probability of it's probability \\(P(w\lvert Context(\omega))\\) can change to choose the right path from root the the leaf node, each node is a binary classification. Suppose code \\(0\\) is a positive label, \\(1\\) is negative label. If the probability of a positive classification is
\\[\sigma(X^T\_\omega \theta)=\frac{1}{1+e^{-X^T\_\omega}}\\]

Then the probability of negative classification is
\\[1-\sigma(X^T\_\omega \theta)\\]

<img src="/images/doc2vec_hierarchical_softmax.png" alt="doc2vec_hierarchical_softmax.png" width="400" />
足球's Huffman code is \\(1001\\), then it's probability in each node are

\\[\begin{aligned}
p(d\_2^\omega\lvert X\_\omega,\theta^\omega\_1&=1-\sigma(X^T\_\omega \theta^\omega\_1))\\\\\\
p(d^\omega\_3\lvert X\_\omega,\theta^\omega\_2&=\sigma(X^T\_\omega \theta^\omega\_2))\\\\\\
p(d^\omega\_4\lvert X\_\omega,\theta^\omega\_3&=\sigma(X^T\_\omega \theta^\omega\_3))\\\\\\
p(d^\omega\_5\lvert X\_\omega,\theta^\omega\_4&=1-\sigma(X^T\_\omega \theta^\omega\_4))\\\\\\
\end{aligned}\\]

where \\(\theta\\) is parameter in the node.

The probability of the `足球` is the production of these equation.

Generally,

\\[p(\omega\lvert Context(\omega))=\prod\_{j=2}^{l\omega}p(d^\omega\_j\lvert X\_\omega,\theta^\omega\_{j-1})\\]


## Negative Sampling {#negative-sampling}

Choose some negative sample, add the probability of the negative word into loss function. Maximize the positive words' probability and minimize the negative words' probability.

Let \\(P(D=0 \lvert \omega,c)\\) be the probability that \\((\omega,c)\\) did not come from the corpus data. Then the objective function will be

\\[\theta = \text{argmax} \prod\_{(\omega,c)\in D} P(D=1\lvert \omega,c,\theta) \prod\_{(\omega,c)\in \tilde{D}} P(D=0\lvert \omega,c,\theta)\\]

where \\(\theta\\) is the parameters of the model(\\(\upsilon\\) and \\(u\\)).

Ref:

-   [word2vec原理推导与代码分析](<http://www.hankcs.com/nlp/word2vec.html>)
-   [CS 224D: Deep Learning for NLP Lecture Notes: Part I](<http://cs224d.stanford.edu/lecture%5Fnotes/notes1.pdf>)
-   [word2vec 中的数学原理详解（一）目录和前言](<http://blog.csdn.net/itplus/article/details/37969519>)

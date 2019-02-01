+++
title = "Models and Architechtures in Word2vec"
author = ["kk"]
date = 2018-01-05T15:14:00+08:00
tags = ["machine learning", "word2vec"]
draft = false
weight = 3003
noauthor = true
nocomment = true
nodate = true
nopaging = true
noread = true
[menu.home]
  weight = 3003
  identifier = "models-and-architechtures-in-word2vec"
+++

## Models {#models}


### CBOW (Continuous Bag of Words) {#cbow--continuous-bag-of-words}

Use the context to predict the probability of current word.
![](/images/doc2vec_cbow.png)

1.  Context words' vectors are \\(\upsilon\_{c-n} ... \upsilon\_{c+m}\\) (\\(m\\) is the window size)
2.  Context vector $ \hat{\upsilon}=\frac{&upsilon;<sub>c-m</sub>+&upsilon;<sub>c-m+1</sub>+...+&upsilon;<sub>c+m</sub>}{2m} $
3.  Score vector \\(z\_i = u\_i\hat{\upsilon}\\), where \\(u\_i\\) is the output vector representation of word \\(\omega\_i\\)
4.  Turn scores into probabilities \\(\hat{y}=softmax(z)\\)
5.  We desire probabilities \\(\hat{y}\\) match the true probabilities \\(y\\).

We use cross entropy \\(H(\hat{y},y)\\) to measure the distance between these two distributions.
\\[H(\hat{y},y)=-\sum\_{j=1}^{\lvert V \rvert}{y\_j\log(\hat{y}\_j)}\\]

\\(y\\) and \\(\hat{y}\\) is accurate, so the loss simplifies to:
\\[H(\hat{y},y)=-y\_j\log(\hat{y})\\]

For perfect prediction, \\(H(\hat{y},y)=-1\log(1)=0\\)

According to this, we can create this loss function:

\\[\begin{aligned}
minimize\ J &=-\log P(\omega\_c\lvert \omega\_{c-m},...,\omega\_{c-1},...,\omega\_{c+m}) \\\\\\
&= -\log P(u\_c \lvert \hat{\upsilon}) \\\\\\
&= -\log \frac{\exp(u\_c^T\hat{\upsilon})}{\sum\_{j=1}^{\lvert V \rvert}\exp (u\_j^T\hat{\upsilon})} \\\\\\
&= -u\_c^T\hat{\upsilon}+\log \sum\_{j=1}^{\lvert V \rvert}\exp (u\_j^T\hat{\upsilon})
\end{aligned}\\]


### Skip-Gram {#skip-gram}

Use current word to predict its context.
![](/images/doc2vec_skip-gram.png)

1.  We get the input word's vector \\(\upsilon\_c\\)
2.  Generate \\(2m\\) score vectors, \\(uc\_{c-m},...,u\_{c-1},...,u\_{c+m}\\).
3.  Turn scores into probabilities \\(\hat{y}=softmax(u)\\)
4.  We desire probabilities \\(\hat{y}\\) match the true probabilities \\(y\\).

\\[\begin{aligned}
minimize J &=-\log P(\omega\_{c-m},...,\omega\_{c-1},\omega\_{c+1},...\omega\_{c+m}\lvert \omega\_c)\\\\\\
&=-\log \prod\_{j=0,j\ne m}^{2m}P(\omega\_{c-m+j}\lvert \omega\_c)\\\\\\
&=-\log \prod\_{j=0,j\ne m}^{2m}P(u\_{c-m+j}\lvert \upsilon\_c)\\\\\\
&=-\log \prod\_{j=0,j\ne m}^{2m}\frac{\exp (u^T\_{c-m+j}\upsilon\_c)}{\sum\_{k=1}^{\lvert V \rvert}{\exp (u^T\_k \upsilon\_c)}}\\\\\\
&=-\sum\_{j=0,j\ne m}^{2m}{u^T\_{c-m+j}\upsilon\_c+2m\log \sum\_{k=1}^{\lvert V \rvert} \exp(u^T\_k \upsilon\_c)}
\end{aligned}\\]


## Models {#models}

Minimize \\(J\\) is expensive, as the summation is over \\(\lvert V \rvert\\). There are two ways to reduce the computation. Hierarchical Softmax and Negative Sampling.


### Hierarchical Softmax {#hierarchical-softmax}

Encode words into a huffman tree, then each word has a Huffman code. The probability of it's probability \\(P(w\lvert Context(\omega))\\) can change to choose the right path from root the che leaf node, each node is a binary classification. Suppose code \\(0\\) is a possitive label, \\(1\\) is negative label. If the probability of a possitive classification is
\\[\sigma(X^T\_\omega \theta)=\frac{1}{1+e^{-X^T\_\omega}}\\]

Then the probability of negative classification is
\\[1-\sigma(X^T\_\omega \theta)\\]
![](/images/doc2vec_hierarchical_softmax.png)
足球's Huffman code is \\(1001\\), then it's probability in each node are

\\[\begin{aligned}
p(d\_2^\omega\lvert X\_\omega,\theta^\omega\_1&=1-\sigma(X^T\_\omega \theta^\omega\_1))\\\\\\
p(d^\omega\_3\lvert X\_\omega,\theta^\omega\_2&=\sigma(X^T\_\omega \theta^\omega\_2))\\\\\\
p(d^\omega\_4\lvert X\_\omega,\theta^\omega\_3&=\sigma(X^T\_\omega \theta^\omega\_3))\\\\\\
p(d^\omega\_5\lvert X\_\omega,\theta^\omega\_4&=1-\sigma(X^T\_\omega \theta^\omega\_4))\\\\\\
\end{aligned}\\]

where \\(\theta\\) is prarameter in the node.

The probability of the `足球` is the production of these equation.

Generally,

\\[p(\omega\lvert Context(\omega))=\prod\_{j=2}^{l\omega}p(d^\omega\_j\lvert X\_\omega,\theta^\omega\_{j-1})\\]


### Negative Sampling {#negative-sampling}

Choose some negitive sample, add the probability of the negative word into loss function. Maximize the positive words' probability and minimize the negitive words' probability.

Let \\(P(D=0 \lvert \omega,c)\\) be the probability that \\((\omega,c)\\) did not come from the corpus data. Then the objective funtion will be

\\[\theta = \text{argmax} \prod\_{(\omega,c)\in D} P(D=1\lvert \omega,c,\theta) \prod\_{(\omega,c)\in \tilde{D}} P(D=0\lvert \omega,c,\theta)\\]

where \\(\theta\\) is the parameters of the model(\\(\upsilon\\) and \\(u\\)).

Ref:

-   [word2vec原理推导与代码分析](<http://www.hankcs.com/nlp/word2vec.html>)
-   [CS 224D: Deep Learning for NLP Lecture Notes: Part I](<http://cs224d.stanford.edu/lecture%5Fnotes/notes1.pdf>)
-   [word2vec 中的数学原理详解（一）目录和前言](<http://blog.csdn.net/itplus/article/details/37969519>)

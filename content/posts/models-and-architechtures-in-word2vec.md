+++
title = "Models and Architectures in Word2vec"
author = ["KK"]
date = 2018-01-05T15:14:00+08:00
lastmod = 2020-04-04T23:14:59+08:00
tags = ["Machine Learning", "word2vec"]
draft = false
noauthor = true
nocomment = true
nodate = true
nopaging = true
noread = true
+++

Generally, `word2vec` is a language model to predict the words probability based on the context. When build the model, it create word embedding for each word, and word embedding is widely used in many NLP tasks.


## Models {#models}


### CBOW (Continuous Bag of Words) {#cbow--continuous-bag-of-words}

Use the context to predict the probability of current word. (In the picture, the word is encoded with one-hot encoding, \\(W\_{V\*N}\\) is word embedding, and \\(W\_{V\*N}^{'}\\), the output weight matrix in hidden layer, is same as \\(\hat{\upsilon}\\) in following equations)

{{< figure src="/images/doc2vec_cbow.png" width="400" >}}

1.  Context words' vectors are \\(\upsilon\_{c-n} ... \upsilon\_{c+m}\\) (\\(m\\) is the window size)
2.  Context vector \\(\hat{\upsilon}=\frac{\upsilon\_{c-m}+\upsilon\_{c-m+1}+...+\upsilon\_{c+m}}{2m}\\)
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

{{< figure src="/images/doc2vec_skip-gram.png" width="400" >}}

1.  We get the input word's vector \\(\upsilon\_c\\)
2.  Generate \\(2m\\) score vectors, \\(u\_{c-m},...,u\_{c-1},...,u\_{c+m}\\).
3.  Turn scores into probabilities \\(\hat{y}=softmax(u)\\)
4.  We desire probabilities \\(\hat{y}\\) match the true probabilities \\(y\\).

\\[\begin{aligned}
minimize J &=-\log P(\omega\_{c-m},...,\omega\_{c-1},\omega\_{c+1},...\omega\_{c+m}\lvert \omega\_c)\\\\\\
&=-\log \prod\_{j=0,j\ne m}^{2m}P(\omega\_{c-m+j}\lvert \omega\_c)\\\\\\
&=-\log \prod\_{j=0,j\ne m}^{2m}P(u\_{c-m+j}\lvert \upsilon\_c)\\\\\\
&=-\log \prod\_{j=0,j\ne m}^{2m}\frac{\exp (u^T\_{c-m+j}\upsilon\_c)}{\sum\_{k=1}^{\lvert V \rvert}{\exp (u^T\_k \upsilon\_c)}}\\\\\\
&=-\sum\_{j=0,j\ne m}^{2m}{u^T\_{c-m+j}\upsilon\_c+2m\log \sum\_{k=1}^{\lvert V \rvert} \exp(u^T\_k \upsilon\_c)}
\end{aligned}\\]


## Architectures {#architectures}

Minimize \\(J\\) is expensive, you need to calculate the probability of each word in vocabulary list. There are two ways to reduce the computation. Hierarchical Softmax and Negative Sampling.


### Hierarchical Softmax {#hierarchical-softmax}

Encode words into a huffman tree, then each word has a Huffman code. The probability of it's probability \\(P(w\lvert Context(\omega))\\) can change to choose the path from root to the leaf node, each node is a binary classification. Suppose code \\(0\\) is a positive label, \\(1\\) is negative label. If the probability of a positive classification is
\\[\sigma(X^T\_\omega \theta)=\frac{1}{1+e^{-X^T\_\omega}}\\]

Then the probability of negative classification is
\\[1-\sigma(X^T\_\omega \theta)\\]

<img src="/images/doc2vec_hierarchical_softmax.png" alt="doc2vec_hierarchical_softmax.png" width="400" />
`足球`'s Huffman code is \\(1001\\), then it's probability in each node are

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

This reduce the calculation complexity to \\(log(n)\\) instead of \\(n\\)


### Negative Sampling {#negative-sampling}

This method will choose some negative sample, then add the probability of the negative word into loss function. The optimisation target becomes maximise the positive words' probability and minimise the negative words' probability.

Let \\(P(D=0 \lvert \omega,c)\\) be the probability that \\((\omega,c)\\) did not come from the corpus data. Then the objective function will be

\\[\theta = \text{argmax} \prod\_{(\omega,c)\in D} P(D=1\lvert \omega,c,\theta) \prod\_{(\omega,c)\in \tilde{D}} P(D=0\lvert \omega,c,\theta)\\]

where \\(\theta\\) is the parameters of the model(\\(\upsilon\\) and \\(u\\)).

---

-   update 04-04-20

I found this two articles pretty useful: [Language Models, Word2Vec, and Efficient Softmax Approximations](https://rohanvarma.me/Word2Vec/) and [Word2vec from Scratch with NumPy](https://towardsdatascience.com/word2vec-from-scratch-with-numpy-8786ddd49e72).


## Ref: {#ref}

1.  [word2vec原理推导与代码分析](<http://www.hankcs.com/nlp/word2vec.html>)
2.  [CS 224D: Deep Learning for NLP Lecture Notes: Part I](<http://cs224d.stanford.edu/lecture%5Fnotes/notes1.pdf>)
3.  [word2vec 中的数学原理详解（一）目录和前言](<http://blog.csdn.net/itplus/article/details/37969519>)

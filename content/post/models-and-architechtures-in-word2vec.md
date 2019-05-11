+++
title = "Machine Learning"
author = ["KK"]
lastmod = 2019-05-11T22:36:00+08:00
tags = ["machine learning", "LSTM", "GRU"]
draft = false
noauthor = true
nocomment = true
nodate = true
nopaging = true
noread = true
+++

## <span class="org-todo done DONE">DONE</span> Models and Architectures in Word2vec {#models-and-architectures-in-word2vec}


### Models {#models}


#### CBOW (Continuous Bag of Words) {#cbow--continuous-bag-of-words}

Use the context to predict the probability of current word.

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


#### Skip-Gram {#skip-gram}

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


### Architectures {#architectures}

Minimize \\(J\\) is expensive, as the summation is over \\(\lvert V \rvert\\). There are two ways to reduce the computation. Hierarchical Softmax and Negative Sampling.

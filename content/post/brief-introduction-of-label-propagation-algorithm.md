+++
title = "Machine Learning"
author = ["KK"]
lastmod = 2019-05-11T22:36:00+08:00
tags = ["machine learning"]
draft = false
noauthor = true
nocomment = true
nodate = true
nopaging = true
noread = true
+++

## <span class="org-todo done DONE">DONE</span> Brief Introduction of Label Propagation Algorithm {#brief-introduction-of-label-propagation-algorithm}

As I said before, I'm working on a text classification project. I use `doc2vec` to convert text into vectors, then I use LPA to classify the vectors.

LPA is a simple, effective semi-supervised algorithm. It can use the density of unlabeled data to find a hyperplane to split the data.

Here are the main stop of the algorithm:

1.  Let $ (x\_1,y1)...(x\_l,y\_l)$ be labeled data, $Y\_L = \\{y\_1...y\_l\\} $ are the class labels. Let \\((x\_{l+1},y\_{l+u})\\) be unlabeled data where \\(Y\_U = \{y\_{l+1}...y\_{l+u}\}\\) are unobserved, usually \\(l \ll u\\). Let \\(X=\{x\_1...x\_{l+u}\}\\) where \\(x\_i\in R^D\\). The problem is to estimate \\(Y\_U\\) for \\(X\\) and \\(Y\_L\\).
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


## <span class="org-todo done DONE">DONE</span> LSTM and GRU {#lstm-and-gru}


### LSTM {#lstm}

The avoid the problem of vanishing gradient and exploding gradient in vanilla RNN, LSTM was published, which can remember information for longer periods of time.

Here is the structure of LSTM:

{{< figure src="/images/LSTM_LSTM.png" width="400" >}}

The calculate procedure are:

\\[\begin{aligned}
f\_t&=\sigma(W\_f\cdot[h\_{t-1},x\_t]+b\_f)\\\\\\
i\_t&=\sigma(W\_i\cdot[h\_{t-1},x\_t]+b\_i)\\\\\\
o\_t&=\sigma(W\_o\cdot[h\_{t-1},x\_t]+b\_o)\\\\\\
\tilde{C\_t}&=tanh(W\_C\cdot[h\_{t-1},x\_t]+b\_C)\\\\\\
C\_t&=f\_t\ast C\_{t-1}+i\_t\ast \tilde{C\_t}\\\\\\
h\_t&=o\_t \ast tanh(C\_t)
\end{aligned}\\]

\\(f\_t\\),\\(i\_t\\),\\(o\_t\\) are forget gate, input gate and output gate respectively. \\(\tilde{C\_t}\\) is the new memory content. \\(C\_t\\) is cell state. \\(h\_t\\) is the output.

Use \\(f\_t\\) and \\(i\_t\\) to update \\(C\_t\\), use \\(o\_t\\) to decide which part of hidden state should be outputted.


### GRU {#gru}

{{< figure src="/images/LSTM_GRU.png" width="400" >}}

\\[\begin{aligned}
z\_t&=\sigma(W\_z\cdot[h\_{t-1},x\_t])\\\\\\
r\_t&=\sigma(W\_r\cdot[h\_{t-1},x\_t])\\\\\\
\tilde{h\_t}&=tanh(W\cdot[r\_t \ast h\_{t-1},x\_t])\\\\\\
h\_t&=(1-z\_t)\ast h\_{t-1}+z\_t \ast \tilde{h\_t}
\end{aligned}\\]

\\(z\_t\\) is update gate, \\(r\_t\\) is reset gate, \\(\tilde{h\_t}\\) is candidate activation, \\(h\_t\\) is activation.

Compare with LSTM, GRU merge cell state and hidden state to one hidden state, and use \\(z\_t\\) to decide how to update the state rather than \\(f\_t\\) and \\(i\_t\\).

Ref:

1.  [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)


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

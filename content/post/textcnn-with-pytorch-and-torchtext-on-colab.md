+++
title = "Architectures"
author = ["KK"]
lastmod = 2019-05-11T22:36:01+08:00
tags = ["machine learning", "word2vec"]
draft = false
noauthor = true
nocomment = true
nodate = true
nopaging = true
noread = true
+++

#### Hierarchical Softmax {#hierarchical-softmax}

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


#### Negative Sampling {#negative-sampling}

Choose some negative sample, add the probability of the negative word into loss function. Maximize the positive words' probability and minimize the negative words' probability.

Let \\(P(D=0 \lvert \omega,c)\\) be the probability that \\((\omega,c)\\) did not come from the corpus data. Then the objective function will be

\\[\theta = \text{argmax} \prod\_{(\omega,c)\in D} P(D=1\lvert \omega,c,\theta) \prod\_{(\omega,c)\in \tilde{D}} P(D=0\lvert \omega,c,\theta)\\]

where \\(\theta\\) is the parameters of the model(\\(\upsilon\\) and \\(u\\)).

Ref:

-   [word2vec原理推导与代码分析](<http://www.hankcs.com/nlp/word2vec.html>)
-   [CS 224D: Deep Learning for NLP Lecture Notes: Part I](<http://cs224d.stanford.edu/lecture%5Fnotes/notes1.pdf>)
-   [word2vec 中的数学原理详解（一）目录和前言](<http://blog.csdn.net/itplus/article/details/37969519>)


## <span class="org-todo done DONE">DONE</span> Parameters in dov2vec {#parameters-in-dov2vec}

Here are some parameter in `gensim`'s `doc2vec` class.


#### window {#window}

window is the maximum distance between the predicted word and context words used for prediction within a document. It will look behind and ahead.

In `skip-gram` model, if the window size is 2, the training samples will be this:(the blue word is the input word)

{{< figure src="/images/doc2vec_window.png" width="400" >}}


#### min\_count {#min-count}

If the word appears less than this value, it will be skipped


#### sample {#sample}

High frequency word like `the` is useless for training. `sample` is a threshold for deleting these higher-frequency words. The probability of keeping the word \\(w\_i\\) is:

\\[P(w\_i) = (\sqrt{\frac{z(\omega\_i)}{s}} + 1) \cdot \frac{s}{z(\omega\_i)}\\]

where \\(z(w\_i)\\) is the frequency of the word and \\(s\\) is the sample rate.

This is the plot when `sample` is 1e-3.

{{< figure src="/images/doc2vec_negative_sample.png" width="400" >}}


#### negative {#negative}

Usually, when training a neural network, for each training sample, all of the weights in the neural network need to be tweaked. For example, if the word pair is ('fox', 'quick'), then only the word quick's neurons should output 1, and all of the other word neurons should output 0.

But it would takes a lot of time to do this when we have billions of training samples. So, instead of update all of the weight, we random choose a small number of "negative" words (default value is 5) to update the weight.(Update their wight to output 0).

So when dealing with word pair ('fox','quick'), we update quick's weight to output 1, and other 5 random words' weight to output 1.

The probability of selecting word \\(\omega\_i\\) is \\(P(\omega\_i)\\):

\\[P(\omega\_i)=\frac{{f(\omega\_i)}^{{3}/{4}}}{\sum\_{j=0}^{n}\left({f(\omega\_j)}^{{3}/{4}}\right)}\\]

\\(f(\omega\_j)\\) is the frequency of word \\(\omega\_j\\).

Ref:

-   [Word2Vec Tutorial - The Skip-Gram Model](<http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/>)
-   [Word2Vec Tutorial Part 2 - Negative Sampling](<http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/>)


## <span class="org-todo done DONE">DONE</span> Semi-supervised text classification using doc2vec and label spreading {#semi-supervised-text-classification-using-doc2vec-and-label-spreading}

Here is a simple way to classify text without much human effort and get a impressive performance.

It can be divided into two steps:

1.  Get train data by using keyword classification
2.  Generate a more accurate classification model by using doc2vec and label spreading


#### Keyword-based Classification {#keyword-based-classification}

Keyword based classification is a simple but effective method. Extracting the target keyword is a monotonous work. I use this method to automatic extract keyword candidate.

1.  Find some most common words to classify the text.
2.  Use this equation to calculate the score of each word appears in the text.
    \\[ score(i) = \frac{count(i)}{all\\\_count(i)^{0.3}}\\]
    which \\(all\\\_count(i)\\) is the word i's word count in all corpus, and \\(count(i)\\) is the word i's word count in positive corpus.
3.  Check the top words, add it to the final keyword list. Repeat this process.

Finally, we can use the keywords to classify the text and get the train data.


#### Classification by `doc2vec` and Label Spreading {#classification-by-doc2vec-and-label-spreading}

Keyword-based classification sometimes produces the wrong result, as it can't using the semantic information in the text. Fortunately, Google has open sourced `word2vec`, which can be used to produce semantically meaningful word embeddings. Furthermore, sentences can also be converted to vectors by using `doc2vec`. Sentences which has closed meaning also have short vector distance.

So the problem is how to classify these vectors.

1.  Using corpus to train the `doc2vec` model.
2.  Using `doc2vec` model to convert sentence into vector.
3.  Using label spreading algorithm to train a classify model to classify the vectors.

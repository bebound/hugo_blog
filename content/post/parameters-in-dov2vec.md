+++
title = "Parameters in dov2vec"
author = ["kk"]
date = 2017-08-03T15:20:00+08:00
lastmod = 2019-02-14T23:37:42+08:00
tags = ["machine learning", "doc2vec"]
draft = false
noauthor = true
nocomment = true
nodate = true
nopaging = true
noread = true
+++

Here are some parameter in `gensim`'s `doc2vec` class.


## window {#window}

window is the maximum distance between the predicted word and context words used for prediction within a document. It will look behind and ahead.

In `skip-gram` model, if the window size is 2, the training samples will be this:(the blue word is the input word)
![](/images/doc2vec_window.png)


## min\_count {#min-count}

If the word appears less than this value, it will be skipped


## sample {#sample}

High frequency word like `the` is useless for training. `sample` is a threshold for deleting these higher-frequency words. The probability of keeping the word \\(w\_i\\) is:

\\[P(w\_i) = (\sqrt{\frac{z(\omega\_i)}{s}} + 1) \cdot \frac{s}{z(\omega\_i)}\\]

where \\(z(w\_i)\\) is the frequency of the word and \\(s\\) is the sample rate.

This is the plot when `sample` is 1e-3.
![](/images/doc2vec_negative_sample.png)


## negative {#negative}

Usually, when training a neural network, for each training sample, all of the weights in the neural network need to be tweaked. For example, if the word pair is ('fox', 'quick'), then only the word quick's neurons should output 1, and all of the other word neurons should output 0.

But it would takes a lot of time to do this when we have billions of training samples. So, instead of update all of the weight, we random choose a small number of "negative" words (default value is 5) to update the weight.(Update their wight to output 0).

So when dealing with word pair ('fox','quick'), we update quick's weight to output 1, and other 5 random words' wight to output 1.

The probability of selecting word \\(\omega\_i\\) is \\(P(\omega\_i)\\):

\\[P(\omega\_i) = \frac{  {f(\omega\_i)}^{3/4}  }{\sum\_{j=0}^{n}\left(  {f(\omega\_j)}^{3/4} \right) }\\]

\\(f(\omega\_j)\\) is the frequency of word \\(\omega\_j\\).

Ref:

-   [Word2Vec Tutorial - The Skip-Gram Model](<http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/>)
-   [Word2Vec Tutorial Part 2 - Negative Sampling](<http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/>)

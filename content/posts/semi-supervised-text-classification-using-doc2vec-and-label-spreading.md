+++
title = "Semi-supervised text classification using doc2vec and label spreading"
author = ["KK"]
date = 2017-09-10T15:29:00+08:00
lastmod = 2020-03-22T22:05:28+08:00
tags = ["Machine Learning", "doc2vec"]
draft = false
noauthor = true
nocomment = true
nodate = true
nopaging = true
noread = true
+++

Here is a simple way to classify text without much human effort and get a impressive performance.

It can be divided into two steps:

1.  Get train data by using keyword classification
2.  Generate a more accurate classification model by using doc2vec and label spreading


## Keyword-based Classification {#keyword-based-classification}

Keyword based classification is a simple but effective method. Extracting the target keyword is a monotonous work. I use this method to automatic extract keyword candidate.

1.  Find some most common words to classify the text.
2.  Use this equation to calculate the score of each word appears in the text.
    \\[ score(i) = \frac{count(i)}{all\\_count(i)^{0.3}}\\]
    where \\(all\\_count(i)\\) is the word \\(i\\)'s word count in all corpus, and \\(count(i)\\) is the word \\(i\\)'s word count in positive corpus.
3.  Check the top words, add it to the final keyword list. Repeat this process.

Finally, we can use the keywords to classify the text and get the train data.


## Classification by doc2vec and Label Spreading {#classification-by-doc2vec-and-label-spreading}

Keyword-based classification sometimes produces the wrong result, as it can't using the semantic information in the text. Fortunately, Google has open sourced `word2vec`, which can be used to produce semantically meaningful word embeddings. Furthermore, sentences can also be converted to vectors by using `doc2vec`. Sentences which has closed meaning also have short vector distance.

So the problem is how to classify these vectors.

1.  Using corpus to train the `doc2vec` model.
2.  Using `doc2vec` model to convert sentence into vector.
3.  Using label spreading algorithm to train a classify model to classify the vectors.
